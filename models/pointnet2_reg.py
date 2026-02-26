
# class get_model(nn.Module):
#     def __init__(self, normal_channel=True): # EDIT : remove num_class since we only looking for one feature
#         super(get_model, self).__init__()
#         in_channel = 3 if normal_channel else 0
#         self.normal_channel = normal_channel
#         self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]]) # Reduces the cloud to 512 points. Looks at 3 scales: radius 0.1, 0.2, and 0.4. 
#         self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]]) # Reduced further to 128 points. The radius double (0.2, 0.4, 0.8) because the cloud is less dense and we want to see larger shapes. 320 at the input corresponds to the sum of the outputs of sa1 (64 + 128 + 128 = 320).
#         self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True) # Everything that remains is grouped into a single vector. This creates a final vector with 1024 dimensions that represents the entire object.
#         self.fc1 = nn.Linear(1024, 512) # classification from 1024
#         self.bn1 = nn.BatchNorm1d(512)
#         self.drop1 = nn.Dropout(0.4) # “turns off” 40% neurons during training to force the model not to rely too much on certain details and to generalize better.
#         self.fc2 = nn.Linear(512, 256)  # classification from 512
#         self.bn2 = nn.BatchNorm1d(256)
#         self.drop2 = nn.Dropout(0.5) # “turns off” 50% of neurons during training to force the model not to rely too much on certain details and to generalize better.
#         self.fc3 = nn.Linear(256, 1)  # classification from 256  # EDIT : one feature output (patient size in cm)

#     def forward(self, xyz):
#         B, _, _ = xyz.shape  # get batch size
#         if self.normal_channel:
#             norm = xyz[:, 3:, :] 
#             xyz = xyz[:, :3, :] 
#         else:
#             norm = None
#         l1_xyz, l1_points = self.sa1(xyz, norm) # Passage to the first layer. We get 512 points with their characteristics.
#         l2_xyz, l2_points = self.sa2(l1_xyz, l1_points) # Passage to the second layer. We drop to 128 points.
#         l3_xyz, l3_points = self.sa3(l2_xyz, l2_points) # Final pass. We obtain a global vector for each object in the batch.
#         x = l3_points.view(B, 1024) # The data is “flattened” to convert from 3D to a flat (vector) format compatible with linear layers.
#         x = self.drop1(F.relu(self.bn1(self.fc1(x)))) # Linear Layer -> Normalization -> ReLU Activation.
#         x = self.drop2(F.relu(self.bn2(self.fc2(x))))
#         x = self.fc3(x)
#         # x = F.log_softmax(x, -1) # calculate probability for each class. # EDIT : no softmax here

#         return x,l3_points


# class get_loss(nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()

#     def forward(self, pred, target, trans_feat=None):
#         # total_loss = F.nll_loss(pred, target) # Negative Log Likelihood Loss
#         total_loss = F.mse_loss(pred.view(-1), target.view(-1).float()) # EDIT : use of Mean Squared Error
#         return total_loss
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

class get_model(nn.Module):
    def __init__(self, normal_channel=False):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        
        # SOLUTION B : Rayons augmentés (10, 20, 40) pour fémurs non-normalisés
        # On garde strictement la structure PointNet2 MSG
        self.sa1 = PointNetSetAbstractionMsg(512, [10.0, 20.0, 40.0], [16, 32, 128], in_channel,
                                              [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        
        self.sa2 = PointNetSetAbstractionMsg(128, [20.0, 40.0, 80.0], [32, 64, 128], 320,
                                              [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        
        self.fc3 = nn.Linear(256, 1) # Sortie : Taille en mètres (ex: 1.75)

    def forward(self, xyz):
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
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x, l3_points

# LA CLASSE MANQUANTE
class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat=None):
        # MSELoss : Calcule la moyenne des carrés des erreurs
        # pred.view(-1) assure qu'on compare des vecteurs de même dimension
        return F.mse_loss(pred.view(-1), target.view(-1).float())