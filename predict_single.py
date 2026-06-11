import os
import sys
import json
import yaml
import torch
import numpy as np
import importlib
import argparse

# Importer les utilitaires géométriques de ton projet
from data_utils.FemurDataLoader import pc_normalize, farthest_point_sample

def parse_args():
    parser = argparse.ArgumentParser(description="Évaluation d'un fémur unique : Vrai vs Prédit")
    parser.add_argument('--config', type=str, required=True, help="Chemin vers le fichier config.yaml du Job (ex: Jobs/15/config.yaml)")
    parser.add_argument('--model_path', type=str, required=True, help="Chemin vers le fichier best_model.pth (ex: Jobs/15/checkpoints/best_model.pth)")
    parser.add_argument('--mesh', type=str, required=True, help="Chemin vers le fichier .obj du fémur à tester")
    parser.add_argument('--json_path', type=str, default="./data/dataset_PatientSize_augmented.json", help="Chemin vers le fichier JSON contenant les vraies tailles")
    return parser.parse_args()

def load_obj(path):
    """Charge les sommets XYZ d'un fichier .obj"""
    vertices = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(x) for x in line.split()[1:4]])
    return np.array(vertices).astype(np.float32)

def find_true_size(mesh_path, json_path):
    """Recherche la vraie taille (PatientSize) associée au maillage dans le fichier JSON"""
    if not os.path.exists(json_path):
        return None
        
    target_name = os.path.basename(mesh_path)
    
    with open(json_path, 'r') as f:
        database = json.load(f)
        
    for item in database:
        # On compare le nom du fichier pour trouver la correspondance
        if os.path.basename(item['obj_path']) == target_name:
            return item['PatientSize']
            
    # Si le maillage n'est pas trouvé directement, on cherche par ID racine (cas des maillages augmentés)
    root_id = target_name.split('_aug')[0].replace('.obj', '')
    for item in database:
        if root_id in os.path.basename(item['obj_path']):
            return item['PatientSize']
            
    return None

def main():
    args = parse_args()

    # 1. Charger la vraie taille dans le JSON
    true_size = find_true_size(args.mesh, args.json_path)

    # 2. Charger la configuration YAML du Job
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    cfg_model = config['model']
    cfg_data = config['dataset']

    # 3. Charger et préparer le nuage de points du maillage .obj
    if not os.path.exists(args.mesh):
        print(f"Erreur : Le fichier maillage {args.mesh} n'existe pas.")
        return
        
    full_point_set = load_obj(args.mesh)
    num_vertices = full_point_set.shape[0]
    npoints = cfg_data['num_point']
    sampling_method = cfg_data['sampling_method'].lower()

    # Application de l'échantillonnage (FPS ou Random) selon la config du Job
    if sampling_method == 'fps':
        point_set = farthest_point_sample(full_point_set, npoints)
    else:
        should_replace = (num_vertices < npoints)
        indices = np.random.choice(num_vertices, npoints, replace=should_replace)
        point_set = full_point_set[indices, :]

    # Normalisation spatiale du fémur (Calcul du scale_factor 'm')
    point_set_norm, m = pc_normalize(point_set[:, 0:3])

    # Préparation des tenseurs pour PyTorch (Batch size = 1, Canaux au milieu [1, 3, N])
    points_tensor = torch.FloatTensor(point_set_norm).unsqueeze(0).transpose(2, 1)
    scale_tensor = torch.FloatTensor([m]).unsqueeze(0)

    # 4. Instancier l'architecture PointNet++ et charger les poids
    model_mod = importlib.import_module('models.pointnet2_reg')
    model = model_mod.get_model(cfg=cfg_model)
    
    if not os.path.exists(args.model_path):
        print(f"Erreur : Le checkpoint {args.model_path} est introuvable.")
        return
        
    checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 5. Exécution de la prédiction
    with torch.no_grad():
        pred_scaled, _ = model(points_tensor, scale_tensor)
        
    # Dé-normalisation du résultat (Mêmes constantes que ton DataLoader original)
    mean_target = 1.70
    std_target = 0.1
    predicted_size = (pred_scaled.item() * std_target) + mean_target

    # 6. Affichage des résultats comparatifs
    print("\n" + "="*60)
    print("         ÉVALUATION COMPARATIVE SUR FÉMUR UNIQUE")
    print("="*60)
    print(f"Fichier Fémur     : {os.path.basename(args.mesh)}")
    print(f"Méthode d'échant. : {sampling_method.upper()} ({npoints} points)")
    print("-" * 60)
    
    if true_size is not None:
        abs_error = abs(true_size - predicted_size)
        print(f"TAILLE RÉELLE     : {true_size:.4f} m  ({true_size*100:.1f} cm)")
        print(f"TAILLE PRÉDITE    : {predicted_size:.4f} m  ({predicted_size*100:.1f} cm)")
        print("-" * 60)
        print(f"ERREUR ABSOLUE    : {abs_error*100:.2f} cm")
    else:
        print(f"TAILLE RÉELLE     : Non trouvée dans le JSON")
        print(f"TAILLE PRÉDITE    : {predicted_size:.4f} m  ({predicted_size*100:.1f} cm)")
        print("-" * 60)
        print("Note : Impossible de calculer l'erreur (Fémur absent du JSON).")
        
    print("="*60 + "\n")

if __name__ == "__main__":
    main()


# python predict_single.py --config Jobs/94/config.yaml --model_path Jobs/94/checkpoints/best_model.pth --mesh ./data/meshes/40001624_m_57.obj