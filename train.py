import os
import sys
import torch
import numpy as np
import datetime
import importlib
import argparse
import csv
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from data_utils.FemurDataLoader import FemurDataLoader

def parse_args():
    parser = argparse.ArgumentParser('Unified PointNet++ Training')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the yaml config file')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 0. Chargement de la configuration YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    cfg_train = config['train']
    cfg_data = config['dataset']
    cfg_model = config['model']

    # Configuration du GPU visible
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg_train['gpu'])

    # 1. Création des dossiers de log
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    log_folder_name = cfg_train['log_dir'] if cfg_train['log_dir'] else timestr
    exp_dir = Path('./log/regression/').joinpath(log_folder_name)
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    # 2. Data Loading (Piloté entièrement par le YAML)
    train_dataset = FemurDataLoader(
        root=cfg_data['root'], 
        npoint=cfg_data['num_point'], 
        split='train',
        sampling_method=cfg_data['sampling_method'],
        augment=cfg_data['augment']
    )
    val_dataset = FemurDataLoader(
        root=cfg_data['root'], 
        npoint=cfg_data['num_point'], 
        split='val',
        sampling_method=cfg_data['sampling_method'],
        augment=False # Toujours False en validation
    )
    test_dataset = FemurDataLoader(
        root=cfg_data['root'], 
        npoint=cfg_data['num_point'], 
        split='test',
        sampling_method=cfg_data['sampling_method'],
        augment=False # Toujours False en test
    )
    
    trainDataLoader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg_train['batch_size'], shuffle=True, num_workers=4, drop_last=True
    )
    valDataLoader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg_train['batch_size'], shuffle=False, num_workers=4
    )
    testDataLoader = torch.utils.data.DataLoader(
        test_dataset, batch_size=cfg_train['batch_size'], shuffle=False, num_workers=4
    )

    # Sauvegarde des paramètres d'échelle du jeu de données
    mean_val = train_dataset.mean_target
    std_val = train_dataset.std_target

    # 3. Model Loading
    # On importe dynamiquement pointnet2_reg (ou autre si spécifié dans le futur)
    model_mod = importlib.import_module('models.pointnet2_reg')
    device = torch.device('cuda' if torch.cuda.is_available() and not args.use_cpu else 'cpu')
    print(f"Utilisation du device : {device}")

    # Initialisation du modèle en lui injectant la sous-section 'model' du YAML
    classifier = model_mod.get_model(cfg=cfg_model).to(device)
    criterion = model_mod.get_loss().to(device)
    
    # Choix de l'optimiseur depuis le YAML
    if cfg_train['optimizer'].lower() == 'sgd':
        optimizer = torch.optim.SGD(classifier.parameters(), lr=cfg_train['learning_rate'], momentum=0.9)
    else:
        optimizer = torch.optim.Adam(classifier.parameters(), lr=cfg_train['learning_rate'])
    
    # 4. Variables de suivi
    best_val_mae_m = 1e10
    history = {
        'train_mse': [], 'val_mse': [],
        'train_mae_cm': [], 'val_mae_cm': []
    }

    # 5. Boucle d'entraînement
    epochs = cfg_train['epoch']
    for epoch in range(epochs):
        # --- PHASE ENTRAÎNEMENT ---
        classifier.train()
        epoch_train_mse = []
        epoch_train_mae_cm = []
        
        for points, target, scale in tqdm(trainDataLoader, total=len(trainDataLoader), desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            points, target, scale = points.to(device).transpose(2, 1), target.to(device), scale.to(device)
            pred, trans_feat = classifier(points, scale)
            
            loss = criterion(pred.view(-1), target.float().view(-1), trans_feat)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                mae_cm = (torch.abs(pred.view(-1) - target.view(-1)).mean() * std_val * 100)
            
            epoch_train_mse.append(loss.item())
            epoch_train_mae_cm.append(mae_cm.item())
        
        history['train_mse'].append(np.mean(epoch_train_mse))
        history['train_mae_cm'].append(np.mean(epoch_train_mae_cm))

        # --- PHASE VALIDATION ---
        classifier.eval()
        epoch_val_mse = []
        epoch_val_mae_m = [] 
        
        with torch.no_grad():
            for points, target, scale in valDataLoader:
                points, target, scale = points.to(device).transpose(2, 1), target.to(device), scale.to(device)
                pred, trans_feat = classifier(points, scale)
                
                v_loss = criterion(pred.view(-1), target.float().view(-1), trans_feat)
                v_mae_m = (torch.abs(pred.view(-1) - target.view(-1)).mean() * std_val)
                
                epoch_val_mse.append(v_loss.item())
                epoch_val_mae_m.append(v_mae_m.item())
        
        avg_val_mae_m = np.mean(epoch_val_mae_m)
        history['val_mse'].append(np.mean(epoch_val_mse))
        history['val_mae_cm'].append(avg_val_mae_m * 100)

        print(f'Epoch {epoch+1}: Train MSE: {history["train_mse"][-1]:.6f}, Val MAE: {avg_val_mae_m*100:.2f}cm')

        if avg_val_mae_m < best_val_mae_m:
            best_val_mae_m = avg_val_mae_m
            torch.save({'model_state_dict': classifier.state_dict(), 'epoch': epoch}, 
                        str(checkpoints_dir) + '/best_model.pth')
            print(f"--- Modèle sauvegardé ({best_val_mae_m*100:.2f}cm) ---")

    # --- ÉVALUATION FINALE (TEST SET) ---
    print("Entraînement terminé. Évaluation finale sur le Test Set...")
    checkpoint = torch.load(str(checkpoints_dir) + '/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    
    test_abs_errors_cm = [] 
    test_signed_errors_cm = []
    test_true_labels_m = []
    test_results_paths = [] 
    
    with torch.no_grad():
        for i, (points, target, scale) in enumerate(tqdm(testDataLoader, desc="Final Testing")):
            points, target, scale = points.to(device).transpose(2, 1), target.to(device), scale.to(device)
            pred, _ = classifier(points, scale)
            
            # Dé-normalisation pour obtenir les valeurs réelles en mètres
            pred_m = (pred.view(-1) * std_val) + mean_val
            target_m = (target.view(-1) * std_val) + mean_val
            
            # Calcul des erreurs absolues et signées (algébriques) en cm
            signed_err_cm = (pred_m - target_m) * 100
            abs_err_cm = torch.abs(signed_err_cm)
            
            test_abs_errors_cm.extend(abs_err_cm.cpu().numpy().tolist())
            test_signed_errors_cm.extend(signed_err_cm.cpu().numpy().tolist())
            test_true_labels_m.extend(target_m.cpu().numpy().tolist())
            
            batch_start = i * cfg_train['batch_size']
            for j in range(len(abs_err_cm)):
                idx = batch_start + j
                if idx < len(test_dataset.datapath):
                    test_results_paths.append(test_dataset.datapath[idx]['obj_path'])
    
    final_test_mae_cm = np.mean(test_abs_errors_cm)
    best_val_mae_cm = best_val_mae_m * 100 

    # Tri de toutes les erreurs du plus grand au plus petit (basé sur l'erreur absolue)
    all_results = list(zip(test_abs_errors_cm, test_signed_errors_cm, test_true_labels_m, test_results_paths))
    sorted_errors = sorted(all_results, key=lambda x: x[0], reverse=True)
    
    # Écriture de l'intégralité du rapport d'erreurs dans un fichier CSV
    csv_log_file = str(exp_dir) + '/all_test_errors.csv'
    with open(csv_log_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Rank', 'True_Label_m', 'Absolute_Error_cm', 'Signed_Error_cm', 'Mesh_Path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for rank, (abs_err, signed_err, label_m, path) in enumerate(sorted_errors):
            # Formatage du signe de l'erreur (+5.20cm ou -3.10cm)
            sign_str = f"+{signed_err:.4f}" if signed_err >= 0 else f"{signed_err:.4f}"
            
            writer.writerow({
                'Rank': rank + 1,
                'True_Label_m': f"{label_m:.4f}",
                'Absolute_Error_cm': f"{abs_err:.4f}",
                'Signed_Error_cm': sign_str,
                'Mesh_Path': path
            })

    print(f'--- BILAN FINAL ---')
    print(f'Meilleure Validation MAE : {best_val_mae_cm:.2f} cm')
    print(f'Test Final MAE : {final_test_mae_cm:.2f} cm')
    print(f"L'intégralité des erreurs signées a été enregistrée dans : {csv_log_file}")
    
    # 6. GÉNÉRATION DES GRAPHES
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_mse'], label='Train MSE', color='blue', alpha=0.6)
    plt.plot(history['val_mse'], label='Val MSE', color='orange', linewidth=2)
    plt.title('Évolution de la Perte MSE (Normalisée)')
    plt.xlabel('Epochs') ; plt.ylabel('Loss')
    plt.legend() ; plt.grid(True, alpha=0.3)
    plt.savefig(str(exp_dir) + '/plot_mse_loss.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(history['train_mae_cm'], label='Train MAE (cm)', color='blue', alpha=0.6)
    plt.plot(history['val_mae_cm'], label='Val MAE (cm)', color='orange', linewidth=2)
    best_epoch = np.argmin(history['val_mae_cm'])
    plt.annotate(f'Best: {best_val_mae_cm:.2f}cm', xy=(best_epoch, best_val_mae_cm), xytext=(best_epoch, best_val_mae_cm + 2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4), horizontalalignment='center')
    plt.title(f'Évolution de l\'Erreur MAE (Best: {best_val_mae_cm:.2f} cm)')
    plt.xlabel('Epochs') ; plt.ylabel('Erreur (cm)')
    plt.legend() ; plt.grid(True, alpha=0.3)
    plt.savefig(str(exp_dir) + '/plot_mae_precision.png')
    plt.close()
    
    # Fichier récapitulatif des paramètres
    config_file = str(exp_dir) + '/parameters_results.txt'
    with open(config_file, 'w') as f:
        f.write('--- CONFIGURATION YAML ---\n')
        f.write(f'Date : {str(datetime.datetime.now())}\n')
        f.write(f'Points demandés : {cfg_data["num_point"]}\n')
        f.write(f'Méthode d\'échantillonnage : {cfg_data["sampling_method"]}\n')
        f.write(f'Utilisation Augmentation (_aug) : {cfg_data["augment"]}\n')
        f.write(f'Batch Size : {cfg_train["batch_size"]}\n')
        f.write(f'Epochs : {cfg_train["epoch"]}\n')
        f.write(f'Learning Rate : {cfg_train["learning_rate"]}\n')
        f.write(f'Optimizer : {cfg_train["optimizer"]}\n\n')
        f.write(f'--- RÉSULTATS ---\n')
        f.write(f'Best Validation MAE (m) : {best_val_mae_m:.6f} m\n')
        f.write(f'Best Validation MAE (cm) : {best_val_mae_cm:.2f} cm\n')
        f.write(f'Final Test MAE : {final_test_mae_cm:.2f} cm\n')
    
    print(f"Résultats finaux consignés dans {config_file}")

if __name__ == '__main__':
    main()