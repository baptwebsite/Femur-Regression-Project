import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

class get_model(nn.Module):
    def __init__(self, normal_channel=False):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        
        # # SA1
        # self.sa1 = PointNetSetAbstractionMsg(
        #     512, 
        #     [0.05, 0.1, 0.2], 
        #     [16, 32, 64], 
        #     in_channel,
        #     [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        
        # # SA2
        # self.sa2 = PointNetSetAbstractionMsg(
        #     128, 
        #     [0.2, 0.4, 0.8], 
        #     [32, 64, 128], 
        #     320,
        #     [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        
        self.sa1 = PointNetSetAbstractionMsg(
            512, 
            [0.02, 0.05, 0.1],      # Rayons réduits
            [16, 32, 48],           
            in_channel,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]])

        # SA2 : Regroupement intermédiaire
        self.sa2 = PointNetSetAbstractionMsg(
            128, 
            [0.1, 0.2, 0.4],        # Rayons réduits 
            [32, 48, 64],           
            320,
            [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        
        # SA3
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        
        # TÊTE DE RÉGRESSION (1025 car 1024 features + 1 scale)
        self.fc1 = nn.Linear(1025, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, 1) 

    # CORRECTION : Ajout de scale_factor dans les arguments
    def forward(self, xyz, scale_factor=None):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        x = l3_points.view(B, 1024)
        
        # --- INJECTION DU FACTEUR D'ÉCHELLE ---
        if scale_factor is not None:
            scale_factor = scale_factor.view(B, 1)
            x = torch.cat([x, scale_factor], dim=1)
        else:
            # Sécurité au cas où scale_factor n'est pas passé (remplissage par des zéros)
            # Mais avec ton nouveau train.py, on passera toujours le scale.
            device = x.device
            extra = torch.zeros((B, 1)).to(device)
            x = torch.cat([x, extra], dim=1)
        
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        
        x = self.fc3(x)
        return x, l3_points
    

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat=None):
        return F.mse_loss(pred.view(-1), target.view(-1).float())