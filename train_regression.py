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

    # 2. Data Loading (Train, Val, Test)
    train_dataset = FemurDataLoader(root='data', npoint=args.num_point, split='train', process_data=args.process_data)
    val_dataset = FemurDataLoader(root='data', npoint=args.num_point, split='val', process_data=args.process_data)
    test_dataset = FemurDataLoader(root='data', npoint=args.num_point, split='test', process_data=args.process_data)
    
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    valDataLoader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 3. Model Loading
    import models.pointnet2_reg as model_mod
    device = torch.device('cuda' if torch.cuda.is_available() and not args.use_cpu else 'cpu')
    print(f"Utilisation du device : {device}")

    classifier = model_mod.get_model(normal_channel=False).to(device)
    criterion = model_mod.get_loss().to(device)
    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate)
    
    # 4. Variables de suivi 
    best_val_mae = 1e10
    history = {
    'train_mse': [], 'val_mse': [],
    'train_mae': [], 'val_mae': []}

    # Paramètres de normalisation (Doivent être identiques à ceux du DataLoader)
    MEAN_SIZE = 1.70
    STD_SIZE = 0.1

   # Initialisation de l'historique étendu
    history = {
        'train_mse': [], 'val_mse': [],
        'train_mae_cm': [], 'val_mae_cm': []
    }

    for epoch in range(args.epoch):
        # --- PHASE ENTRAÎNEMENT ---
        classifier.train()
        epoch_train_mse = []
        epoch_train_mae_cm = []
        
        for points, target in tqdm(trainDataLoader, total=len(trainDataLoader), desc=f"Epoch {epoch+1}/{args.epoch}"):
            optimizer.zero_grad()
            points, target = points.to(device).transpose(2, 1), target.to(device)
            
            pred, trans_feat = classifier(points)
            
            # 1. MSE Loss (sur valeurs normalisées) pour le backprop
            loss = criterion(pred.view(-1), target.float(), trans_feat)
            loss.backward()
            optimizer.step()
            
            # 2. Calcul du MAE en cm pour le suivi
            with torch.no_grad():
                pred_cm = (pred.view(-1) * train_dataset.std_target) # Erreur relative * std = erreur en cm
                target_cm = (target.view(-1) * train_dataset.std_target)
                mae_cm = torch.abs(pred_cm - target_cm).mean()
            
            epoch_train_mse.append(loss.item())
            epoch_train_mae_cm.append(mae_cm.item())
        
        # Stockage moyennes Train
        history['train_mse'].append(np.mean(epoch_train_mse))
        history['train_mae_cm'].append(np.mean(epoch_train_mae_cm))

        # --- PHASE VALIDATION ---
        classifier.eval()
        epoch_val_mse = []
        epoch_val_mae_cm = []
        
        with torch.no_grad():
            for points, target in valDataLoader:
                points, target = points.to(device).transpose(2, 1), target.to(device)
                pred, trans_feat = classifier(points)
                
                # MSE de validation
                v_loss = criterion(pred.view(-1), target.float(), trans_feat)
                
                # MAE de validation en cm
                pred_cm = (pred.view(-1) * train_dataset.std_target)
                target_cm = (target.view(-1) * train_dataset.std_target)
                v_mae_cm = torch.abs(pred_cm - target_cm).mean()
                
                epoch_val_mse.append(v_loss.item())
                epoch_val_mae_cm.append(v_mae_cm.item())
        
        # Stockage moyennes Val
        avg_val_mae = np.mean(epoch_val_mae_cm)
        history['val_mse'].append(np.mean(epoch_val_mse))
        history['val_mae_cm'].append(avg_val_mae)

        print(f'Epoch {epoch+1}: Train MSE: {history["train_mse"][-1]:.6f}, Val MAE: {avg_val_mae:.2f}cm')

        # Sauvegarde du meilleur modèle basé sur le MAE de validation
        if (avg_val_mae / 100) < best_val_mae:
            best_val_mae = (avg_val_mae / 100)
            torch.save({'model_state_dict': classifier.state_dict(), 'epoch': epoch}, 
                        str(checkpoints_dir) + '/best_model.pth')
            print(f"--- Modèle sauvegardé ({avg_val_mae:.2f}cm) ---")

    # --- ÉVALUATION FINALE SUR LE TEST SET ---
    print("Entraînement terminé. Évaluation finale...")
    checkpoint = torch.load(str(checkpoints_dir) + '/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    
    test_errors_cm = []
    with torch.no_grad():
        for points, target in testDataLoader:
            points, target = points.to(device).transpose(2, 1), target.to(device)
            pred, _ = classifier(points)
            # Conversion directe en cm via l'écart-type
            err_cm = torch.abs(pred.view(-1) - target.view(-1)) * train_dataset.std_target
            test_errors_cm.extend(err_cm.cpu().numpy())
    
    print(f'--- BILAN FINAL ---')
    print(f'RÉSULTAT TEST FINAL -> MAE RÉELLE : {np.mean(test_errors_cm):.2f} cm')

    # --- 6. GÉNÉRATION DES GRAPHES (MSE et MAE séparés) ---
    # Plot 1 : MSE (Train vs Val)
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_mse'], label='Train MSE', color='blue')
    plt.plot(history['val_mse'], label='Val MSE', color='orange', linewidth=2)
    plt.title('Évolution de la Perte MSE (Normalisée)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(str(exp_dir) + '/plot_mse_loss.png')
    plt.close()

    # Plot 2 : MAE (Train vs Val)
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_mae_cm'], label='Train MAE (cm)', color='blue')
    plt.plot(history['val_mae_cm'], label='Val MAE (cm)', color='orange', linewidth=2)
    plt.title('Évolution de l\'Erreur MAE (en cm)')
    plt.xlabel('Epochs')
    plt.ylabel('Erreur (cm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(str(exp_dir) + '/plot_mae_precision.png')
    plt.close()
    
    print(f"Graphiques sauvegardés dans {exp_dir}/")

    # --- SAUVEGARDE DES HYPERPARAMÈTRES et RESULTATS ---
    config_file = str(exp_dir) + '/parameters_results.txt'
    with open(config_file, 'w') as f:
        f.write('--- HYPERPARAMÈTRES DE L\'ENTRAÎNEMENT ---\n')
        f.write(f'Date : {str(datetime.datetime.now())}\n')
        f.write(f'Modèle : {args.model}\n')
        f.write(f'Batch Size : {args.batch_size}\n')
        f.write(f'Points : {args.num_point}\n')
        f.write(f'Epochs : {args.epoch}\n')
        f.write(f'Learning Rate : {args.learning_rate}\n')
        f.write(f'Optimizer : {args.optimizer}\n')
 
        f.write(f'\n--- RÉSULTATS FINAUX ---\n')
        f.write(f'Best Validation MAE (m) : {best_val_mae:.6f}\n')
        f.write(f'Best Validation MAE (cm) : {best_val_mae * 100:.2f} cm\n')
        f.write(f'Final Test MAE (cm) : {np.mean(test_errors_cm):.2f} cm\n')
        f.write(f'Entraînement terminé à : {str(datetime.datetime.now())}\n')

        print(f"Paramètres sauvegardés dans {config_file}")

if __name__ == '__main__':
    args = parse_args()
    main(args)






