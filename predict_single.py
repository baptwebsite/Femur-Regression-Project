import os
import sys
import yaml
import torch
import numpy as np
import importlib
import argparse

# Importer les utilitaires géométriques existants
from data_utils.FemurDataLoader import pc_normalize, farthest_point_sample

def parse_args():
    parser = argparse.ArgumentParser(description="Inférence PointNet++ sur un fémur unique")
    parser.add_argument('--config', type=str, required=True, help="Chemin vers le fichier config.yaml du Job")
    parser.add_argument('--model_path', type=str, required=True, help="Chemin vers le fichier best_model.pth")
    parser.add_argument('--mesh', type=str, required=True, help="Chemin vers le fichier .obj à prédire")
    return parser.parse_args()

def load_obj(path):
    """Charge les sommets XYZ d'un fichier .obj"""
    vertices = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.split()[1:4]])
    return np.array(vertices).astype(np.float32)

def main():
    args = parse_args()

    # 1. Charger la configuration YAML pour reproduire l'architecture exacte
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    cfg_model = config['model']
    cfg_data = config['dataset']

    # 2. Charger et préparer le nuage de points du maillage .obj
    if not os.path.exists(args.mesh):
        print(f"Erreur : Le fichier maillage {args.mesh} n'existe pas.")
        return
        
    print(f"-> Chargement du maillage : {args.mesh}")
    full_point_set = load_obj(args.mesh)
    num_vertices = full_point_set.shape[0]
    npoints = cfg_data['num_point']
    sampling_method = cfg_data['sampling_method'].lower()

    # Appliquer l'échantillonnage configuré (FPS ou Random)
    print(f"-> Application de l'échantillonnage ({sampling_method.upper()}) à {npoints} points...")
    if sampling_method == 'fps':
        point_set = farthest_point_sample(full_point_set, npoints)
    else:
        should_replace = (num_vertices < npoints)
        indices = np.random.choice(num_vertices, npoints, replace=should_replace)
        point_set = full_point_set[indices, :]

    # Normalisation spatiale (Extraction de m)
    point_set_norm, m = pc_normalize(point_set[:, 0:3])

    # Préparation des tenseurs pour PyTorch (Batch dimension = 1)
    # PointNet++ attend la forme [B, C, N], donc on transpose les axes 1 et 2
    points_tensor = torch.FloatTensor(point_set_norm).unsqueeze(0).transpose(2, 1)
    scale_tensor = torch.FloatTensor([m]).unsqueeze(0)

    # 3. Instancier le modèle et charger les poids entraînés
    model_mod = importlib.import_module('models.pointnet2_reg')
    model = model_mod.get_model(cfg=cfg_model)
    
    if not os.path.exists(args.model_path):
        print(f"Erreur : Le checkpoint {args.model_path} est introuvable.")
        return
        
    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 4. Exécuter la prédiction (Inférence)
    print("-> Exécution du modèle...")
    with torch.no_grad():
        pred_scaled, _ = model(points_tensor, scale_tensor)
        
    # 5. Dé-normalisation de la cible (PatientSize)
    # Utilisation des mêmes constantes strictes que le DataLoader
    mean_target = 1.70
    std_target = 0.1
    
    predicted_size_m = (pred_scaled.item() * std_target) + mean_target

    # 6. Affichage du résultat clinique
    print("\n" + "="*50)
    print("               RÉSULTAT DE LA PRÉDICTION")
    print("="*50)
    print(f"Fichier analysé       : {os.path.basename(args.mesh)}")
    print(f"Facteur d'échelle (m) : {m:.4f} (taille d'origine de l'os)")
    print(f"Taille prédite        : {predicted_size_m:.4f} m  ({predicted_size_m*100:.2f} cm)")
    print("="*50)

if __name__ == "__main__":
    main()