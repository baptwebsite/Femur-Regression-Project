"""
Author: Benny (Adapted for Femur Regression)
Date: Jan 2026
"""
from data_utils.FemurDataLoader import FemurDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
from sklearn.metrics import r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in testing')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number') 
    parser.add_argument('--model', type=str, default='pointnet2_reg', help='model name')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate predictions')
    return parser.parse_args()


def test(model, loader, device, mean_target, std_target, vote_num=1):
    classifier = model.eval()
    all_preds_m = []   
    all_targets_m = [] 

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        points, target = points.to(device), target.to(device)
        points = points.transpose(2, 1)
        
        batch_pred_sum = torch.zeros(target.size()[0], 1).to(device)
        
        for _ in range(vote_num):
            pred, _ = classifier(points)
            batch_pred_sum += pred
        
        final_pred_norm = batch_pred_sum / vote_num
        
        # --- DÉ-NORMALISATION ---
        # Utilisation des vraies valeurs mu et std du dataset
        final_pred_m = (final_pred_norm.view(-1) * std_target) + mean_target
        target_m = (target.view(-1) * std_target) + mean_target
        
        all_preds_m.extend(final_pred_m.cpu().numpy())
        all_targets_m.extend(target_m.cpu().numpy())

    all_preds_m = np.array(all_preds_m)
    all_targets_m = np.array(all_targets_m)
    
    diff = all_preds_m - all_targets_m
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    r2 = r2_score(all_targets_m, all_preds_m)
    
    return mae, rmse, r2

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() and not args.use_cpu else 'cpu')

    '''CREATE DIR'''
    experiment_dir = 'log/regression/' + args.log_dir

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
        
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/' 

    # Chargement du dataset
    test_dataset = FemurDataLoader(root=data_path, npoint=args.num_point, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # RÉCUPÉRATION DES VALEURS DE NORMALISATION
    mu = test_dataset.mean_target
    std = test_dataset.std_target
    log_string(f'Normalisation utilisée: Mean={mu:.4f}, Std={std:.4f}')

    '''MODEL LOADING'''
    from models import pointnet2_reg
    classifier = pointnet2_reg.get_model(normal_channel=args.use_normals).to(device)

    # Chargement des poids entraînés
    checkpoint_path = os.path.join(experiment_dir, 'checkpoints/best_model.pth')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])

    log_string('Start Evaluation...')
    with torch.no_grad():
        # On passe mu et std à la fonction test
        mae, rmse, r2 = test(classifier, testDataLoader, device, mu, std, vote_num=args.num_votes)
        
        log_string('\n' + '='*30)
        log_string('--- RÉSULTATS FINAUX (TEST) ---')
        log_string('MAE (Erreur Moyenne Absolue): %.4f m' % mae)
        log_string('MAE en Centimètres: %.2f cm' % (mae * 100))
        log_string('RMSE (Racine de l\'erreur quadratique): %.4f m' % rmse)
        log_string('R² Score: %.4f' % r2)
        log_string('='*30)

if __name__ == '__main__':
    args = parse_args()
    main(args)