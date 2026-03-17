import os
import sys
import torch
import numpy as np
import datetime
import logging
import importlib
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from data_utils.FemurDataLoader import FemurDataLoader

def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--model', default='pointnet2_reg', help='model name')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--process_data', action='store_true', default=True, help='save data offline')
    return parser.parse_args()

def main(args):
    # 1. Création des dossiers de log
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/regression/').joinpath(args.log_dir if args.log_dir else timestr)
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    # 2. Data Loading
    train_dataset = FemurDataLoader(root='data', npoint=args.num_point, split='train', process_data=args.process_data)
    val_dataset = FemurDataLoader(root='data', npoint=args.num_point, split='val', process_data=args.process_data)
    test_dataset = FemurDataLoader(root='data', npoint=args.num_point, split='test', process_data=args.process_data)
    
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    valDataLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 3. Model Loading
    model_name = args.model
    model_mod = importlib.import_module(f'models.{model_name}')
    device = torch.device('cuda' if torch.cuda.is_available() and not args.use_cpu else 'cpu')
    print(f"Utilisation du device : {device}")

    classifier = model_mod.get_model(normal_channel=False).to(device)
    criterion = model_mod.get_loss().to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate)
    
    # 4. Variables de suivi (Tout est géré en METRES en interne)
    best_val_mae_m = 1e10
    history = {
        'train_mse': [], 'val_mse': [],
        'train_mae_cm': [], 'val_mae_cm': []
    }

    # Récupération de l'écart-type pour la dénormalisation
    std_val = train_dataset.std_target

    for epoch in range(args.epoch):
        # --- PHASE ENTRAÎNEMENT ---
        classifier.train()
        epoch_train_mse = []
        epoch_train_mae_cm = []
        
        for points, target in tqdm(trainDataLoader, total=len(trainDataLoader), desc=f"Epoch {epoch+1}/{args.epoch}"):
            optimizer.zero_grad()
            points, target = points.to(device).transpose(2, 1), target.to(device)
            pred, trans_feat = classifier(points)
            
            # MSE Loss sur valeurs normalisées
            loss = criterion(pred.view(-1), target.float(), trans_feat)
            loss.backward()
            optimizer.step()
            
            # MAE en cm pour le suivi historique
            with torch.no_grad():
                # (pred - target) * std = erreur en mètres -> * 100 = cm
                mae_cm = (torch.abs(pred.view(-1) - target.view(-1)).mean() * std_val * 100)
            
            epoch_train_mse.append(loss.item())
            epoch_train_mae_cm.append(mae_cm.item())
        
        history['train_mse'].append(np.mean(epoch_train_mse))
        history['train_mae_cm'].append(np.mean(epoch_train_mae_cm))

        # --- PHASE VALIDATION ---
        classifier.eval()
        epoch_val_mse = []
        epoch_val_mae_m = [] # On stocke en mètres pour la logique de sauvegarde
        
        with torch.no_grad():
            for points, target in valDataLoader:
                points, target = points.to(device).transpose(2, 1), target.to(device)
                pred, trans_feat = classifier(points)
                
                v_loss = criterion(pred.view(-1), target.float(), trans_feat)
                # Erreur en mètres
                v_mae_m = (torch.abs(pred.view(-1) - target.view(-1)).mean() * std_val)
                
                epoch_val_mse.append(v_loss.item())
                epoch_val_mae_m.append(v_mae_m.item())
        
        avg_val_mae_m = np.mean(epoch_val_mae_m)
        history['val_mse'].append(np.mean(epoch_val_mse))
        history['val_mae_cm'].append(avg_val_mae_m * 100)

        print(f'Epoch {epoch+1}: Train MSE: {history["train_mse"][-1]:.6f}, Val MAE: {avg_val_mae_m*100:.2f}cm')

        # Sauvegarde du meilleur modèle (Logique cohérente en mètres)
        if avg_val_mae_m < best_val_mae_m:
            best_val_mae_m = avg_val_mae_m
            torch.save({'model_state_dict': classifier.state_dict(), 'epoch': epoch}, 
                        str(checkpoints_dir) + '/best_model.pth')
            print(f"--- Modèle sauvegardé ({best_val_mae_m*100:.2f}cm) ---")

    # --- ÉVALUATION FINALE ---
    print("Entraînement terminé. Évaluation finale sur le Test Set...")
    checkpoint = torch.load(str(checkpoints_dir) + '/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    
    test_errors_m = []
    std_val = train_dataset.std_target # On récupère l'écart-type pour dénormaliser
    
    with torch.no_grad():
        for points, target in testDataLoader:
            points, target = points.to(device).transpose(2, 1), target.to(device)
            pred, _ = classifier(points)
            
            # Calcul de l'erreur brute en mètres (pred et target sont normalisés)
            err_m = torch.abs(pred.view(-1) - target.view(-1)) * std_val
            test_errors_m.extend(err_m.cpu().numpy().tolist())
    
    final_test_mae_m = np.mean(test_errors_m)
    final_test_mae_cm = final_test_mae_m * 100
    best_val_mae_cm = best_val_mae_m * 100 # best_val_mae_m a été sauvé pendant la boucle

    print(f'--- BILAN FINAL ---')
    print(f'Meilleure Validation MAE : {best_val_mae_cm:.2f} cm')
    print(f'Test Final MAE           : {final_test_mae_cm:.2f} cm')
    
    # 6. GÉNÉRATION DES GRAPHES
    best_mae_cm = best_val_mae_m * 100

    # Plot 1 : MSE
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_mse'], label='Train MSE', color='blue', alpha=0.6)
    plt.plot(history['val_mse'], label='Val MSE', color='orange', linewidth=2)
    plt.title('Évolution de la Perte MSE (Normalisée)')
    plt.xlabel('Epochs') ; plt.ylabel('Loss')
    plt.legend() ; plt.grid(True, alpha=0.3)
    plt.savefig(str(exp_dir) + '/plot_mse_loss.png')
    plt.close()

    # Plot 2 : MAE
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_mae_cm'], label='Train MAE (cm)', color='blue', alpha=0.6)
    plt.plot(history['val_mae_cm'], label='Val MAE (cm)', color='orange', linewidth=2)
    
    best_epoch = np.argmin(history['val_mae_cm'])
    plt.annotate(f'Best: {best_mae_cm:.2f}cm', 
                 xy=(best_epoch, best_mae_cm), 
                 xytext=(best_epoch, best_mae_cm + 2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=4),
                 horizontalalignment='center')

    plt.title(f'Évolution de l\'Erreur MAE (Best: {best_mae_cm:.2f} cm)')
    plt.xlabel('Epochs') ; plt.ylabel('Erreur (cm)')
    plt.legend() ; plt.grid(True, alpha=0.3)
    plt.savefig(str(exp_dir) + '/plot_mae_precision.png')
    plt.close()
    
    # --- SAUVEGARDE DES RESULTATS ---
    config_file = str(exp_dir) + '/parameters_results.txt'
    with open(config_file, 'w') as f:
        f.write('--- CONFIGURATION ET RÉSULTATS ---\n')
        f.write(f'Date : {str(datetime.datetime.now())}\n')
        f.write(f'Modèle : {args.model} | Points : {args.num_point}\n')
        f.write(f'Batch Size : {args.batch_size} | Epochs : {args.epoch}\n')
        f.write(f'Learning Rate : {args.learning_rate} | Optimizer : {args.optimizer}\n\n')
        f.write(f'--- RÉSULTATS ---\n')
        f.write(f'Best Validation MAE (m) : {best_val_mae_m:.6f} m\n')
        f.write(f'Best Validation MAE (cm) : {best_mae_cm:.2f} cm\n')
        f.write(f'Final Test MAE (cm) : {final_test_mae_cm:.2f} cm\n')
    
    print(f"Entraînement terminé. Résultats dans {config_file}")

if __name__ == '__main__':
    args = parse_args()
    main(args)