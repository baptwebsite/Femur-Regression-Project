import os
import numpy as np
import warnings
import pickle
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

def pc_normalize(pc):
    """
    Normalisation pour la régression :
    On centre le nuage de points sur son centroïde, mais on NE divise PAS par 
    le rayon max pour conserver l'échelle absolue du fémur.
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    # m = np.max(np.sqrt(np.sum(pc**2, axis=1))) # Supprimé pour garder l'échelle
    # pc = pc / m 
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

        
        
        # 1. Chargement du JSON
        json_path = os.path.join(self.root, 'dataset_PatientSize.json')
        with open(json_path, 'r') as f:
            all_data = json.load(f)

        # 2. Split ternaire (Train: 70%, Val: 15%, Test: 15%)
        # On sépare d'abord le Test (15%)
        train_val_data, test_data = train_test_split(all_data, test_size=0.15, random_state=42)
        # On sépare le Train et le Val (0.18 * 0.85 ≈ 0.15 du total)
        train_data, val_data = train_test_split(train_val_data, test_size=0.18, random_state=42)

        if split == 'train':
            self.datapath = train_data
        elif split == 'val':
            self.datapath = val_data
        else:
            self.datapath = test_data
        
        print('The size of %s data is %d' % (split, len(self.datapath)))

        # Nom du cache spécifique au split pour éviter les erreurs de chargement
        self.save_path = os.path.join(root, 'femur_%s_%dpts.dat' % (split, self.npoints))
        
        if self.process_data:
            if not os.path.exists(self.save_path):
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    item = self.datapath[index]
                    # Gestion des chemins Windows/Linux
                    rel_path = item['obj_path'].replace('\\', os.sep)
                    obj_path = os.path.join(self.root, rel_path)
                    
                    point_set = self.load_obj(obj_path)
                    label = np.array([item['PatientSize']]).astype(np.float32)
                    
                    # Sous-échantillonnage FPS
                    point_set = farthest_point_sample(point_set, self.npoints)

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = label

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)


        self.mean_target = 1.70
        self.std_target = 0.1


    def load_obj(self, path):
        vertices = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertices.append([float(x) for x in line.split()[1:4]])
        return np.array(vertices).astype(np.float32)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            item = self.datapath[index]
            rel_path = item['obj_path'].replace('\\', os.sep)
            obj_path = os.path.join(self.root, rel_path)
            point_set = self.load_obj(obj_path)
            label = np.array([item['PatientSize']]).astype(np.float32)
            point_set = farthest_point_sample(point_set, self.npoints)
        
        # Normalisation (Centrage uniquement)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        # --- AJOUT : Normalisation de la Cible (Target Scaling) ---
        # On centre sur 1.70m et on divise par 0.1 pour amplifier les écarts
        target = label[0]
        target_scaled = (target - self.mean_target) / self.std_target
        # ----------------------------------------------------------
        
        # Retourne le nuage de points et la taille (normalisé).
        return point_set, target_scaled

    def __getitem__(self, index):
        return self._get_item(index)