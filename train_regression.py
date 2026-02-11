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
    history = {'train_loss': [], 'val_mae': []}

    # Paramètres de normalisation (Doivent être identiques à ceux du DataLoader)
    MEAN_SIZE = 1.70
    STD_SIZE = 0.1

    for epoch in range(args.epoch):
        # PHASE ENTRAÎNEMENT
        classifier.train()
        train_loss_epoch = []
        for points, target in tqdm(trainDataLoader, total=len(trainDataLoader), desc=f"Epoch {epoch+1}/{args.epoch}"):
            optimizer.zero_grad()
            points, target = points.to(device).transpose(2, 1), target.to(device)
            
            pred, trans_feat = classifier(points)
            
            # Le critère calcule la distance sur les valeurs normalisées (ex: entre 0.2 et 0.5)
            loss = criterion(pred.view(-1), target.float(), trans_feat)
            loss.backward()
            optimizer.step()
            train_loss_epoch.append(loss.item())
        
        avg_train_loss = np.mean(train_loss_epoch)
        history['train_loss'].append(avg_train_loss)

        # PHASE VALIDATION
        classifier.eval()
        val_errors_cm = [] # On stocke les erreurs en CM
        with torch.no_grad():
            for points, target in valDataLoader:
                points, target = points.to(device).transpose(2, 1), target.to(device)
                pred, _ = classifier(points)
                
                # --- INVERSION DE LA NORMALISATION ---
                # On repasse les prédictions et les cibles en mètres pour calculer la MAE réelle
                pred_m = (pred.view(-1) * STD_SIZE) + MEAN_SIZE
                target_m = (target.view(-1) * STD_SIZE) + MEAN_SIZE
                
                # Calcul de l'erreur absolue en centimètres
                e_cm = torch.abs(pred_m - target_m) * 100
                val_errors_cm.extend(e_cm.cpu().numpy())
        
        avg_val_mae_cm = np.mean(val_errors_cm)
        # On stocke la MAE en mètres (pour la cohérence du best_val_mae)
        history['val_mae'].append(avg_val_mae_cm / 100)
        
        print(f'Epoch {epoch+1}: Loss: {avg_train_loss:.6f}, Val MAE: {avg_val_mae_cm:.2f}cm')

        # Sauvegarde si amélioration de la MAE (en cm ou m, le résultat est le même)
        if (avg_val_mae_cm / 100) < best_val_mae:
            best_val_mae = (avg_val_mae_cm / 100)
            torch.save({'model_state_dict': classifier.state_dict(), 'epoch': epoch}, 
                    str(checkpoints_dir) + '/best_model.pth')
            print("--- Modèle sauvegardé (Meilleure MAE Val) ---")

    # # 5. ÉVALUATION FINALE SUR LE TEST SET
    # print("Entraînement terminé. Évaluation finale sur le Test Set...")
    # checkpoint = torch.load(str(checkpoints_dir) + '/best_model.pth')
    # classifier.load_state_dict(checkpoint['model_state_dict'])
    # classifier.eval()
    # test_errors = []
    # with torch.no_grad():
    #     for points, target in testDataLoader:
    #         points, target = points.to(device).transpose(2, 1), target.to(device)
    #         pred, _ = classifier(points)
    #         test_errors.extend(torch.abs(pred.view(-1) - target.view(-1)).cpu().numpy())
    
    # print(f'RÉSULTAT TEST FINAL -> MAE: {np.mean(test_errors)*100:.2f}cm')

    # 5. ÉVALUATION FINALE SUR LE TEST SET
    print("Entraînement terminé. Évaluation finale sur le Test Set...")
    checkpoint = torch.load(str(checkpoints_dir) + '/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()
    
    # On récupère les paramètres de normalisation calculés par le dataset
    # (Assure-toi que ton train_dataset est toujours accessible ici)
    mu = train_dataset.mean_target
    std = train_dataset.std_target
    
    test_errors_normalized = []
    
    with torch.no_grad():
        for points, target in testDataLoader:
            points, target = points.to(device).transpose(2, 1), target.to(device)
            pred, _ = classifier(points)
            
            # On stocke l'erreur normalisée
            error_norm = torch.abs(pred.view(-1) - target.view(-1))
            test_errors_normalized.extend(error_norm.cpu().numpy())
    
    # CALCUL DE LA VRAIE MAE :
    # Erreur réelle = Erreur_normalisée * Écart-type
    mae_norm = np.mean(test_errors_normalized)
    mae_reelle_cm = mae_norm * std
    
    print(f'--- BILAN FINAL ---')
    print(f'MAE Normalisée : {mae_norm:.4f}')
    print(f'RÉSULTAT TEST FINAL -> MAE RÉELLE : {mae_reelle_cm:.2f} cm')
    
    # 6. GÉNÉRATION DES GRAPHES
    plt.figure(figsize=(12, 5))
    
    # Graphe de la perte
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train MSE Loss')
    plt.title('Perte d\'entraînement')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Graphe de la MAE
    plt.subplot(1, 2, 2)
    plt.plot([x * 100 for x in history['val_mae']], label='Val MAE (cm)', color='orange')
    plt.title('Précision (Validation)')
    plt.xlabel('Epochs')
    plt.ylabel('Erreur (cm)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(str(exp_dir) + '/history_plot.png')
    print(f"Graphes sauvegardés dans {exp_dir}/history_plot.png")

if __name__ == '__main__':
    args = parse_args()
    main(args)