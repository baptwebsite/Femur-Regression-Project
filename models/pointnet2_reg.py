import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

class get_model(nn.Module):
    def __init__(self, normal_channel=False):
        super(get_model, self).__init__()
        # Si normal_channel est True, on a 6 canaux (XYZ + Vecteurs normaux), sinon 3 (XYZ uniquement)
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        
        # PREMIER NIVEAU D'ABSTRACTION (SA1) 
        # On réduit les 2048 points d'entrée à 512 points.
        # [10.0, 20.0, 40.0] : Rayons de recherche pour capturer des détails locaux à différentes échelles (MSG).
        # [16, 32, 128] : Nombre de points voisins consultés pour chaque rayon.
        # [[32, 32, 64], ...] : Architectures des MLP pour chaque échelle.
        self.sa1 = PointNetSetAbstractionMsg(
            512, 
            [0.05, 0.1, 0.2], 
            [16, 32, 64], 
            in_channel,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        
        # DEUXIÈME NIVEAU D'ABSTRACTION (SA2)
        # On réduit encore les 512 points à 128 points pour capturer des formes plus globales.
        # 320 : Nombre de canaux en entrée (somme des sorties de sa1 : 64 + 128 + 128).
        # Les rayons sont plus grands [20.0, 40.0, 80.0] car le nuage de points est plus clairsemé.
        self.sa2 = PointNetSetAbstractionMsg(
            128, 
            [0.2, 0.4, 0.8], 
            [32, 64, 128], 
            320,
            [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        
        # ABSTRACTION GLOBALE (SA3)
        # Ici, on ne garde plus de points (None), on compresse tout le nuage en un seul vecteur global.
        # Cela crée un descripteur unique de 1024 dimensions qui représente la forme entière du fémur.
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        
        # TÊTE DE RÉGRESSION (FULLY CONNECTED)
        # On passe du vecteur de 1024 à la valeur finale via des couches denses.
        
        # Couche 1 : 1024 -> 512 neurones
        self.fc1 = nn.Linear(1025, 512)
        self.bn1 = nn.BatchNorm1d(512) # Normalisation pour stabiliser l'apprentissage
        self.drop1 = nn.Dropout(0.4)    # Désactive 40% des neurones 
        
        # Couche 2 : 512 -> 256 neurones
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)    # Désactive 50% des neurones
        
        # Couche 3 : Sortie finale (1 seul neurone)
        # Prédit la valeur scalaire (la longueur du fémur en mètres)
        self.fc3 = nn.Linear(256, 1) 

    def forward(self, xyz):
        B, _, _ = xyz.shape
        # Gestion des canaux normaux si présents
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        # Passage à travers les couches d'abstraction successives
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Préparation du vecteur global pour les couches denses (Flatten)
        x = l3_points.view(B, 1024)
        
       # --- INJECTION DU FACTEUR D'ÉCHELLE ---
        if scale_factor is not None:
            # On s'assure que scale_factor a la forme (B, 1)
            scale_factor = scale_factor.view(B, 1)
            # Concaténation : le vecteur devient (B, 1025)
            x = torch.cat([x, scale_factor], dim=1)
        
        # Passage dans les couches denses
        # ATTENTION : Ton self.fc1 doit être défini avec in_features=1025 dans __init__
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        
        x = self.fc3(x)
        return x, l3_points
    

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat=None):
        # MSELoss : Calcule la moyenne des carrés des erreurs
        # pred.view(-1) assure qu'on compare des vecteurs de même dimension
        return F.mse_loss(pred.view(-1), target.view(-1).float())