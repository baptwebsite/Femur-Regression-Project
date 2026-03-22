import os
import numpy as np
import warnings
import pickle
import json
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

def pc_normalize(pc):
    """ Centre le nuage de points sur son centroïde. """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    return pc

def farthest_point_sample(point, npoint):
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
        
        # Paramètres de normalisation de la cible
        self.mean_target = 1.70
        self.std_target = 0.1

        # 1. Chargement du JSON 
        json_path = os.path.join(self.root, 'dataset_PatientSize_augmented.json')
        with open(json_path, 'r') as f:
            all_data = json.load(f)

        # 2. LOGIQUE DE SPLIT (Groupement par ID Racine)
        groups = defaultdict(list)
        for item in all_data:
            # On extrait l'ID de base : meshes\40000037_m_50_aug1.obj -> 40000037_m_50
            filename = os.path.basename(item['obj_path'])
            root_id = filename.split('_aug')[0].replace('.obj', '')
            groups[root_id].append(item)
        
        # On split sur les IDs uniques (les patients réels)
        unique_root_ids = sorted(list(groups.keys()))
        
        # --- CONFIGURATION 80 / 10 / 10 ---
        
        # 1. On isole 10% pour le Test (reste 90% pour Train+Val)
        train_val_ids, test_ids = train_test_split(
            unique_root_ids, 
            test_size=0.10, 
            random_state=42
        )
        
        # 2. On veut que Val représente 10% du TOTAL.
        # Comme il reste 90% des données, on prend 1/9ème de ce reste :
        # 0.10 / 0.90 ≈ 0.1111
        train_ids, val_ids = train_test_split(
            train_val_ids, 
            test_size=0.1111, 
            random_state=42
        )

        # Attribution des données selon le split demandé
        if split == 'train':
            selected_ids = train_ids
        elif split == 'val':
            selected_ids = val_ids
        else:
            selected_ids = test_ids
            
        self.datapath = [entry for rid in selected_ids for entry in groups[rid]]
        
        print(f'--- Split {split} ---')
        print(f'Nombre de patients uniques : {len(selected_ids)}')
        print(f'Nombre total de maillages (avec augmentations) : {len(self.datapath)}')

        # Gestion du cache
        self.save_path = os.path.join(root, f'femur_{split}_{self.npoints}pts.dat')
        
        if self.process_data:
            if not os.path.exists(self.save_path):
                self.list_of_points = []
                self.list_of_labels = []

                for item in tqdm(self.datapath, desc=f"Processing {split}"):
                    rel_path = item['obj_path'].replace('\\', os.sep)
                    obj_path = os.path.join(self.root, rel_path)
                    
                    point_set = self.load_obj(obj_path)
                    label = np.array([item['PatientSize']]).astype(np.float32)
                    
                    # Sous-échantillonnage FPS
                    point_set = farthest_point_sample(point_set, self.npoints)

                    self.list_of_points.append(point_set)
                    self.list_of_labels.append(label)

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print(f'Load processed data from {self.save_path}...')
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def load_obj(self, path):
        vertices = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append([float(x) for x in line.split()[1:4]])
        return np.array(vertices).astype(np.float32)

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            item = self.datapath[index]
            rel_path = item['obj_path'].replace('\\', os.sep)
            obj_path = os.path.join(self.root, rel_path)
            point_set = self.load_obj(obj_path)
            label = np.array([item['PatientSize']]).astype(np.float32)
            point_set = farthest_point_sample(point_set, self.npoints)
        
        # Normalisation spatiale
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # Normalisation de la cible (Standard Scaling)
        target = label[0]
        target_scaled = (target - self.mean_target) / self.std_target
        
        return point_set, target_scaled
    
    def __getitem__(self, index):
        return self._get_item(index)