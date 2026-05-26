import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

class get_model(nn.Module):
    def __init__(self, cfg):
        """
        Initialise le modèle PointNet++ Regression entièrement à partir du dictionnaire de configuration YAML.
        cfg: correspond à la section 'model' du fichier YAML.
        """
        super(get_model, self).__init__()
        
        # 1. Configuration des canaux d'entrée
        self.normal_channel = cfg.get('normal_channel', False)
        in_channel = 3 if self.normal_channel else 0
        
        # 2. Couche SA1 (Abstraction locale multi-échelle 1)
        self.sa1 = PointNetSetAbstractionMsg(
            cfg['sa1']['npoint'], 
            cfg['sa1']['radii'], 
            cfg['sa1']['nsample'], 
            in_channel,
            cfg['sa1']['mlp']
        )
        
        # Calcul dynamique du canal d'entrée de SA2 (somme des derniers filtres de chaque MLP de SA1)
        sa2_in_channel = sum([mlp_branches[-1] for mlp_branches in cfg['sa1']['mlp']])
        
        # 3. Couche SA2 (Abstraction locale multi-échelle 2)
        self.sa2 = PointNetSetAbstractionMsg(
            cfg['sa2']['npoint'], 
            cfg['sa2']['radii'], 
            cfg['sa2']['nsample'], 
            sa2_in_channel,
            cfg['sa2']['mlp']
        )
        
        # Calcul dynamique du canal d'entrée de SA3 (dernier filtre du dernier bloc MLP de SA2 + 3 pour XYZ)
        sa3_in_channel = sum([mlp_branches[-1] for mlp_branches in cfg['sa2']['mlp']]) + 3
        
        # 4. Couche SA3 (Abstraction globale)
        self.sa3 = PointNetSetAbstraction(
            npoint=None, 
            radius=None, 
            nsample=None, 
            in_channel=sa3_in_channel, 
            mlp=cfg['sa3']['mlp'], 
            group_all=True
        )
        
        # 5. Tête de Régression Linéaire Évolutive
        # La sortie globale de SA3 correspond au dernier élément de sa liste MLP, auquel on ajoute +1 pour le scale_factor
        final_feature_dim = cfg['sa3']['mlp'][-1]
        fc1_input_dim = final_feature_dim + 1 
        
        # Récupération des hyperparamètres de la tête depuis le YAML
        reg_cfg = cfg['regression_head']
        
        self.fc1 = nn.Linear(fc1_input_dim, reg_cfg['fc1_units'])
        self.bn1 = nn.BatchNorm1d(reg_cfg['fc1_units'])
        self.drop1 = nn.Dropout(reg_cfg['dropout_1'])
        
        self.fc2 = nn.Linear(reg_cfg['fc1_units'], reg_cfg['fc2_units'])
        self.bn2 = nn.BatchNorm1d(reg_cfg['fc2_units'])
        self.drop2 = nn.Dropout(reg_cfg['dropout_2'])
        
        # Couche finale de sortie (toujours 1 pour la régression scalaire de PatientSize)
        self.fc3 = nn.Linear(reg_cfg['fc2_units'], 1) 

    def forward(self, xyz, scale_factor=None):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        # Descente dans l'architecture PointNet++
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Redimensionnement dynamique basé sur les caractéristiques extraites de SA3
        x = l3_points.view(B, -1)
        
        # --- INJECTION DU FACTEUR D'ÉCHELLE (SCALE) ---
        if scale_factor is not None:
            scale_factor = scale_factor.view(B, 1)
            x = torch.cat([x, scale_factor], dim=1)
        else:
            device = x.device
            extra = torch.zeros((B, 1)).to(device)
            x = torch.cat([x, extra], dim=1)
        
        # Passage dans les blocs Fully Connected
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        
        x = self.fc3(x)
        return x, l3_points
    

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat=None):
        return F.mse_loss(pred.view(-1), target.view(-1).float())