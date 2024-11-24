from torch.nn import Module
from models.diffusion import DiffusionPoint, PointCloudUNet, VarianceSchedule
from models import CLIP_TEXT_DIM

class GaussianVAE(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        emb_dim = CLIP_TEXT_DIM[args.clip_version]
        self.diffusion = DiffusionPoint(
            net = PointCloudUNet(point_dim=args.point_dim, num_heads=args.num_heads, time_dim=1280, context_dim=emb_dim),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            ),
            time_emb=320
        )
        
    def get_loss(self, x, context):
        """
        Args:
            x:  Input point clouds, (B, C, N).
        """
        

        loss = self.diffusion.get_loss(x, context)

        return loss

    def sample(self, num_points, context, flexibility, truncate_std=None):
        """
        Args:
            z:  Input latent, normal random samples with mean=0 std=1, (B, F)
        """
        if truncate_std is not None:
            z = truncated_normal_(z, mean=0, std=1, trunc_std=truncate_std)
        samples = self.diffusion.sample(num_points, context=context, point_dim=args.point_dim, flexibility=flexibility)
        return samples


if __name__ == '__main__':
    import torch
    import argparse

    parser = argparse.ArgumentParser()
    # Model arguments
    parser.add_argument('--point_dim', type=int, default=9)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=100)
    parser.add_argument('--beta_1', type=float, default=1e-4)
    parser.add_argument('--beta_T', type=float, default=0.02)
    parser.add_argument('--sched_mode', type=str, default='linear')
    parser.add_argument('--flexibility', type=float, default=0.0)
    parser.add_argument('--truncate_std', type=float, default=2.0)
    parser.add_argument('--latent_flow_depth', type=int, default=14)
    parser.add_argument('--latent_flow_hidden_dim', type=int, default=256)
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--sample_num_points', type=int, default=2048)
    parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
    parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
    parser.add_argument('--clip_version', type=str, default="ViT-H-14")
    parser.add_argument('--clip_pretrained', type=str, default="laion2b_s32b_b79k")  
    
    args = parser.parse_args()
    network = GaussianVAE(args)


    B = 4
    N = 2048
    pcd = torch.rand((B, args.point_dim, N)).cuda()
    context = torch.randn((B, 77, 1024)).cuda()

    network = network.cuda()
    pcd = pcd.cuda()

    loss = network.get_loss(pcd, context)

    print(loss)
    
    network.eval()
    pcd = network.sample(num_points=1024, context=context, flexibility=0)
    
    print(pcd)
