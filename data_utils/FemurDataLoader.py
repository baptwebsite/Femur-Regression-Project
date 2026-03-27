import os
import numpy as np
import warnings
import pickle
import json
import torch
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

def pc_normalize(pc):
    """
    Centre le nuage de points et le ramène dans une sphère de rayon 1.
    Retourne le nuage normalisé et le facteur d'échelle original.
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    # Calcul de la distance la plus lointaine (norme L2 max)
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    # Éviter la division par zéro au cas où
    if m > 0:
        pc = pc / m
    return pc, m

def farthest_point_sample(point, npoint):
    """
    Échantillonnage par points les plus éloignés (FPS).
    """
    N, D = point.shape
    xyz = point[:,:3]
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
    def __init__(self, root, npoint=2048, split='train', process_data=False):
        self.root = root 
        self.npoints = npoint
        self.process_data = process_data
        self.split = split
        
        # Paramètres de normalisation de la cible (moyenne et std du PatientSize)
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
        print(f'Nombre de maillages chargés : {len(self.datapath)}')

        # 4. Gestion du cache (Fichier .dat)
        self.save_path = os.path.join(root, f'femur_{split}_{self.npoints}pts_v2.dat')
        
        if self.process_data:
            if not os.path.exists(self.save_path):
                self.list_of_points = []
                self.list_of_labels = []

                for item in tqdm(self.datapath, desc=f"Preprocessing {split}"):
                    rel_path = item['obj_path'].replace('\\', os.sep)
                    obj_path = os.path.join(self.root, rel_path)
                    
                    point_set = self.load_obj(obj_path)
                    label = np.array([item['PatientSize']]).astype(np.float32)
                    
                    # Sous-échantillonnage FPS au préalable pour le cache
                    point_set = farthest_point_sample(point_set, self.npoints)

                    self.list_of_points.append(point_set)
                    self.list_of_labels.append(label)

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print(f'Chargement des données pré-traitées depuis {self.save_path}...')
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

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
        # Récupération des données brutes
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            item = self.datapath[index]
            rel_path = item['obj_path'].replace('\\', os.sep)
            obj_path = os.path.join(self.root, rel_path)
            point_set = self.load_obj(obj_path)
            label = np.array([item['PatientSize']]).astype(np.float32)
            point_set = farthest_point_sample(point_set, self.npoints)
        
        # 1. Normalisation spatiale complète
        # point_set_norm est dans une sphère de rayon 1, m est l'échelle originale
        point_set_norm, m = pc_normalize(point_set[:, 0:3])

        # 2. Normalisation de la cible (Regression Target)
        target = label[0]
        target_scaled = (target - self.mean_target) / self.std_target
        
        # 3. Retourne (Points, Cible, Facteur d'échelle)
        # On convertit m en tableau numpy pour que le DataLoader le gère en batch
        scale_factor = np.array([m]).astype(np.float32)
        
        return point_set_norm, target_scaled, scale_factor