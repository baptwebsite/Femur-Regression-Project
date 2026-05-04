import os
import numpy as np
import warnings
import json
import torch
from collections import defaultdict
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

def pc_normalize(pc):
    """
    Centre le nuage de points et le ramène dans une sphère de rayon 1.
    Retourne le nuage normalisé et le facteur d'échelle original (m).
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    if m > 0:
        pc = pc / m
    return pc, m

class FemurDataLoader(Dataset):
    def __init__(self, root, npoint=2048, split='train', process_data=False):
        self.root = root 
        self.npoints = npoint
        self.split = split
        
        # Paramètres de normalisation de la cible (PatientSize)
        self.mean_target = 1.70
        self.std_target = 0.1

        # 1. Chargement du fichier JSON d'indexation
        json_path = os.path.join(self.root, 'dataset_PatientSize_augmented.json')
        with open(json_path, 'r') as f:
            all_data = json.load(f)

        # 2. Logique de Split par Patient (ID Racine) pour éviter les fuites
        groups = defaultdict(list)
        for item in all_data:
            filename = os.path.basename(item['obj_path'])
            # meshes/40000037_m_50_aug1.obj -> root_id = 40000037_m_50
            root_id = filename.split('_aug')[0].replace('.obj', '')
            groups[root_id].append(item)
        
        unique_root_ids = sorted(list(groups.keys()))
        
        # Split 80% Train / 10% Val / 10% Test
        train_val_ids, test_ids = train_test_split(
            unique_root_ids, 
            test_size=0.10, 
            random_state=42
        )
        
        train_ids, val_ids = train_test_split(
            train_val_ids, 
            test_size=0.1111, # 0.1111 * 0.90 ≈ 0.10 du total
            random_state=42
        )

        # 3. Sélection des données selon le split
        if split == 'train':
            selected_ids = train_ids
            self.datapath = [entry for rid in selected_ids for entry in groups[rid]]
        elif split == 'val':
            selected_ids = val_ids
            self.datapath = [entry for rid in selected_ids for entry in groups[rid]]
        else:
            # Pour le TEST : On exclut strictement les données augmentées
            selected_ids = test_ids
            self.datapath = [
                entry for rid in selected_ids 
                for entry in groups[rid] 
                if '_aug' not in entry['obj_path']
            ]
            
        print(f'--- Initialisation Dataset Femur [{split}] ---')
        print(f'Nombre de patients uniques : {len(selected_ids)}')
        print(f'Nombre de maillages : {len(self.datapath)}')

    def load_obj(self, path):
        """ Charge les sommets d'un fichier .obj """
        vertices = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append([float(x) for x in line.split()[1:4]])
        return np.array(vertices).astype(np.float32)

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        # 1. Lecture directe du fichier sur le disque
        item = self.datapath[index]
        rel_path = item['obj_path'].replace('\\', os.sep)
        obj_path = os.path.join(self.root, rel_path)
        
        # On charge tous les points du fémur
        full_point_set = self.load_obj(obj_path)
        label = np.array([item['PatientSize']]).astype(np.float32)

        # 2. Échantillonnage ALÉATOIRE DYNAMIQUE
        # On pioche npoints au hasard à chaque appel (chaque epoch)
        num_vertices = full_point_set.shape[0]
        selected_indices = np.random.choice(
            num_vertices, 
            self.npoints, 
            replace=(num_vertices < self.npoints)
        )
        point_set = full_point_set[selected_indices, :]
        
        # 3. Normalisation spatiale XYZ
        point_set_norm, m = pc_normalize(point_set)

        # 4. Normalisation de la cible
        target = label[0]
        target_scaled = np.array((target - self.mean_target) / self.std_target).astype(np.float32)
        
        # 5. Facteur d'échelle (m)
        scale_factor = np.array([m]).astype(np.float32)
        
        return point_set_norm, target_scaled, scale_factor