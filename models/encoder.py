import torch
import torch.nn.functional as F
from torch import nn
from models.pointnet2_utils import PointNetSetAbstractionMsg

class PointNet2Encoder(nn.Module):
    def __init__(self):
        super(PointNet2Encoder, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 9, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(512, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(256, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(64, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 4], [256, 384, 4]])
        
    def forward(self, xyz, noise):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
                
        mean, log_variance = torch.chunk(l4_points, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()
        x = mean + stdev * noise
        x *= 0.18215
        return x, l4_xyz



if __name__ == '__main__':
    import  torch
    model = PointNet2Encoder().cuda()
    xyz = torch.rand(1, 9, 1024).cuda()
    noise = torch.rand(1, 64).cuda()
    model(xyz, noise)

