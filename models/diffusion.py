import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, MultiheadAttention, Conv1d, Dropout, BatchNorm1d, Sequential, Linear
import numpy as np
from models.common import ConcatSquashLinear, AttentionBlock
from models.pointnet2_utils import PointNetFeaturePropagation, PointNetSetAbstractionMsg
from tqdm.auto import tqdm

class TimeEmbedding(Module):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = Linear(n_embd, 4 * n_embd)
        self.linear_2 = Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x

class VarianceSchedule(Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas
    
    def get_time_embedding(self, timestep, dtype):
        freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=dtype) / 160)
        x = torch.tensor([timestep], dtype=dtype)[:, None] * freqs[None]
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

        
class SwitchSequential(Sequential):
    def forward(self, point_pos0, point_pos1, point_feat0, point_feat1, context, time):
        for layer in self:
            if isinstance(layer, AttentionBlock):
                point_feat0 = layer(point_feat0, context)
            elif isinstance(layer, PointNetSetAbstractionMsg):
                point_pos0, point_feat0 = layer(point_pos0, point_feat0, time)
            elif isinstance(layer, PointNetFeaturePropagation):
                point_feat0 = layer(point_pos0, point_pos1, point_feat0, point_feat1, time)
                point_pos0 = None
            else:
                raise NotImplementedError("unknown layer.", type(layer))
        return point_pos0, point_feat0
        
        
class PointCloudUNet(Module):
    def __init__(self, point_dim=9, num_heads=4, time_dim=320, context_dim=768):
        super(PointCloudUNet, self).__init__()
        self.encoders = ModuleList([
            SwitchSequential(PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], point_dim, [[16, 16, 32], [32, 32, 64]], time_dim), AttentionBlock(num_heads, 96, context_dim)),
            SwitchSequential(PointNetSetAbstractionMsg(512, [0.1, 0.2], [16, 32], 32+64, [[64, 64, 128], [64, 96, 128]], time_dim), AttentionBlock(num_heads, 256, context_dim)),
            SwitchSequential(PointNetSetAbstractionMsg(256, [0.2, 0.4], [16, 32], 128+128, [[128, 196, 256], [128, 196, 256]], time_dim), AttentionBlock(num_heads, 512, context_dim)),
            SwitchSequential(PointNetSetAbstractionMsg(64, [0.4, 0.8], [16, 32], 256+256, [[256, 256, 512], [256, 384, 512]], time_dim), AttentionBlock(num_heads, 1024, context_dim)),
        ])
        
        self.decoders = ModuleList([
            SwitchSequential(PointNetFeaturePropagation(512+512+256+256, [256, 256], time_dim), AttentionBlock(num_heads, 256, context_dim)),
            SwitchSequential(PointNetFeaturePropagation(128+128+256, [256, 256], time_dim), AttentionBlock(num_heads, 256, context_dim)),
            SwitchSequential(PointNetFeaturePropagation(32+64+256, [256, 128], time_dim), AttentionBlock(num_heads, 128, context_dim)),
            SwitchSequential(PointNetFeaturePropagation(point_dim+128, [128, 128, 128], time_dim), AttentionBlock(num_heads, 128, context_dim)),
        ])
              
        
        self.conv1 = Conv1d(128, 128, 1)
        self.bn1 = BatchNorm1d(128)
        self.drop1 = Dropout(0.5)
        self.conv2 = Conv1d(128, point_dim, 1)
        
    def forward(self, x, context, time):
        xyzs, points = [x[:,:3,:]], [x]
        for i, layers in enumerate(self.encoders):
            xyz, point = layers(xyzs[i], None, points[i], None, context, time)
            xyzs.append(xyz)
            points.append(point)
        
        xyz = xyzs.pop()
        point = points.pop()
        
        for i, layers in enumerate(self.decoders):
            xyz_before = xyzs.pop()
            point_before = points.pop()
                                    
            _, point = layers(xyz_before, xyz, point_before, point, context, time)
            xyz = xyz_before
        
        x = self.drop1(F.relu(self.bn1(self.conv1(point))))
        x = self.conv2(x)
        return x
        
        
        


class DiffusionPoint(Module):

    def __init__(self, net, time_emb, var_sched: VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched
        self.time_embedding = TimeEmbedding(time_emb)

    def get_loss(self, x, context, t=None):
        """
        Args:
            x: Input point cloud, (B, C, N).
            context: text embeddings (B, text_len, text_dim).
        """
        batch_size, point_dim, _ = x.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
            time_embedding = torch.cat([self.var_sched.get_time_embedding(t_, x.dtype) for t_ in t], dim=0).to(x)
            time_embedding = self.time_embedding(time_embedding)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)   # (B, 1, 1)

        e_rand = torch.randn_like(x)  # (B, C, N)
        e_theta = self.net(c0 * x + c1 * e_rand, context=context, time=time_embedding)

        loss = F.mse_loss(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim), reduction='mean')
        return loss

    def sample(self, num_points, context, point_dim=3, flexibility=0.0, ret_traj=False):
        batch_size = context.size(0)
        x_T = torch.randn([batch_size, point_dim, num_points]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        for t in tqdm(range(self.var_sched.num_steps, 0, -1), desc="Denoising..."):
            z = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)
            time_emb = self.var_sched.get_time_embedding(t, x_T.dtype).to(x_T)
            time_emb = self.time_embedding(time_emb)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / torch.sqrt(alpha)
            c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t] * batch_size]
            e_theta = self.net(x_t, context=context, time=time_emb)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t - 1] = x_next.detach()  # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()  # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj
        else:
            return traj[0]


if __name__ == "__main__":
    time_emb = 320
    B = 4
    N = 2048
    C = 3
    
    
    unet = PointCloudUNet(C).cuda()
    
    diffusion = DiffusionPoint(unet,)
        
    xyz = torch.randn((B, C, N)).cuda()
    time_emb = torch.randn((B, time_emb)).cuda()
    
    text_features = torch.randn((B, 77, 768)).cuda()
        
    y = unet(xyz, text_features, time_emb)
    
    print(y.shape)
