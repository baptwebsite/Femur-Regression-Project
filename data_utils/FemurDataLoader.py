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

def farthest_point_sample(point, npoint):
    """
    Échantillonnage par points les plus éloignés (FPS).
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class FemurDataLoader(Dataset):
    def __init__(self, root, npoint=2048, split='train', sampling_method='random', augment=False):
        self.root = root 
        self.npoints = npoint
        self.split = split
        self.sampling_method = sampling_method.lower()
        
        # L'augmentation par fichiers physiques ne s'applique que sur le train set
        self.augment = augment if split == 'train' else False
        
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

        # 3. Sélection et filtrage des données selon le split et l'argument augment
        if split == 'train':
            selected_ids = train_ids
            if not self.augment:
                # Exclure les données augmentées si augment=False
                self.datapath = [
                    entry for rid in selected_ids 
                    for entry in groups[rid] 
                    if '_aug' not in entry['obj_path']
                ]
            else:
                # Inclure tout (base + augmentés) si augment=True
                self.datapath = [entry for rid in selected_ids for entry in groups[rid]]
                
        elif split == 'val':
            selected_ids = val_ids
            # Mode validation : on exclut toujours les données augmentées pour garder des métriques réelles
            self.datapath = [
                entry for rid in selected_ids 
                for entry in groups[rid] 
                if '_aug' not in entry['obj_path']
            ]
        else:
            selected_ids = test_ids
            # Mode test : On exclut strictement les données augmentées
            self.datapath = [
                entry for rid in selected_ids 
                for entry in groups[rid] 
                if '_aug' not in entry['obj_path']
            ]
            
        print(f'--- Initialisation Dataset Femur [{split}] ---')
        print(f'Mode d\'échantillonnage : {self.sampling_method.upper()}')
        print(f'Utilisation de l\'augmentation (_aug) : {self.augment}')
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

    def _sample_points(self, full_point_set):
        """ Applique dynamiquement la méthode d'échantillonnage configurée """
        num_vertices = full_point_set.shape[0]
        
        if self.sampling_method == 'fps':
            return farthest_point_sample(full_point_set, self.npoints)
        else:
            # Mode 'random' par défaut sans remplacement (avec sécurité dynamique)
            should_replace = (num_vertices < self.npoints)
            selected_indices = np.random.choice(
                num_vertices, 
                self.npoints, 
                replace=should_replace
            )
            return full_point_set[selected_indices, :]

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        # Lecture directe à la volée depuis le disque
        item = self.datapath[index]
        rel_path = item['obj_path'].replace('\\', os.sep)
        obj_path = os.path.join(self.root, rel_path)
        
        full_point_set = self.load_obj(obj_path)
        label = np.array([item['PatientSize']]).astype(np.float32)
        
        # Échantillonnage dynamique (recalculé à chaque itération/epoch si mode random)
        point_set = self._sample_points(full_point_set)
        
        # 1. Normalisation spatiale XYZ
        point_set_norm, m = pc_normalize(point_set[:, 0:3])

        # 2. Normalisation de la cible
        target = label[0]
        target_scaled = np.array((target - self.mean_target) / self.std_target).astype(np.float32)
        
        # 3. Facteur d'échelle original emballé pour PyTorch
        scale_factor = np.array([m]).astype(np.float32)
        
        return point_set_norm, target_scaled, scale_factor