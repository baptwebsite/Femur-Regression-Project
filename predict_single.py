import os
import torch
import numpy as np
import argparse
import importlib

from data_utils.FemurDataLoader import pc_normalize, farthest_point_sample, FemurDataLoader

def parse_args():
    parser = argparse.ArgumentParser('predict')
    parser.add_argument('--model_path', type=str, required=True, help='Chemin du .pth')
    parser.add_argument('--mesh_path', type=str, required=True, help='Chemin du fichier .obj')
    parser.add_argument('--target', type=float, required=True, help='Taille réelle attendue (m)')
    parser.add_argument('--num_point', type=int, default=2048, help='Nombre de points')
    parser.add_argument('--model_name', default='pointnet2_reg', help='Nom du modèle')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Chargement du modèle
    model_mod = importlib.import_module(f'models.{args.model_name}')
    classifier = model_mod.get_model(normal_channel=False).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier.eval()

    # 2. Prétraitement avec les fonctions importées
    loader_utils = FemurDataLoader(root='data', npoint=args.num_point, split='test', process_data=False)
    
    print(f"Chargement et échantillonnage de {args.mesh_path}...")
    raw_points = loader_utils.load_obj(args.mesh_path)
    sampled_points = farthest_point_sample(raw_points, args.num_point)
    
    # Normalisation (Centrage)
    norm_points = pc_normalize(sampled_points)
    
    # Préparation du tenseur
    input_tensor = torch.from_numpy(norm_points).float().unsqueeze(0).transpose(2, 1).to(device)

    # 3. Inférence et Dénormalisation
    with torch.no_grad():
        pred, _ = classifier(input_tensor)
        pred_scaled = pred.item()
        
        predicted_size_m = (pred_scaled * loader_utils.std_target) + loader_utils.mean_target

    # 4. Affichage
    error_cm = abs(predicted_size_m - args.target) * 100
    
    print("\n" + "="*40)
    print(f"RÉSULTAT DE L'INFÉRENCE (FONCTIONS IMPORTÉES)")
    print("="*40)
    print(f"Prédiction : {predicted_size_m:.4f} m")
    print(f"Réel       : {args.target:.4f} m")
    print(f"Erreur     : {error_cm:.2f} cm")
    print("="*40 + "\n")

if __name__ == '__main__':
    main()