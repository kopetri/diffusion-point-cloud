from torch.nn import Module
from models.encoders import PointNetEncoder
from models.diffusion import DiffusionPoint, PointwiseNet, VarianceSchedule
from models.common import reparameterize_gaussian, standard_normal_logprob, gaussian_entropy, truncated_normal_
from models import CLIP_TEXT_DIM

class GaussianVAE(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        emb_dim = CLIP_TEXT_DIM[args.clip_version]
        self.encoder = PointNetEncoder(args.latent_dim)
        self.diffusion = DiffusionPoint(
            net = PointwiseNet(point_dim=args.latent_dim, context_dim=args.latent_dim, text_dim=emb_dim),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )
        
    def get_loss(self, x, writer=None, it=None, kl_weight=1.0, text_embeddings=None):
        """
        Args:
            x:  Input point clouds, (B, N, d).
        """
        batch_size, _, _ = x.size()
        z_mu, z_sigma, point_emb = self.encoder(x)
        z = reparameterize_gaussian(mean=z_mu, logvar=z_sigma)  # (B, F)
        log_pz = standard_normal_logprob(z).sum(dim=1)  # (B, ), Independence assumption
        entropy = gaussian_entropy(logvar=z_sigma)      # (B, )
        loss_prior = (- log_pz - entropy).mean()

        loss_recons = self.diffusion.get_loss(x, point_emb, z, text_embeddings=text_embeddings)

        loss = kl_weight * loss_prior + loss_recons

        if writer is not None:
            writer.add_scalar('train/loss_entropy', -entropy.mean(), it)
            writer.add_scalar('train/loss_prior', -log_pz.mean(), it)
            writer.add_scalar('train/loss_recons', loss_recons, it)

        return loss

    def sample(self, z, num_points, flexibility, truncate_std=None, text_embeddings=None):
        """
        Args:
            z:  Input latent, normal random samples with mean=0 std=1, (B, F)
        """
        if truncate_std is not None:
            z = truncated_normal_(z, mean=0, std=1, trunc_std=truncate_std)
        samples = self.diffusion.sample(num_points, context=z, flexibility=flexibility, text_embeddings=text_embeddings)
        return samples


if __name__ == '__main__':
    import torch
    import argparse

    parser = argparse.ArgumentParser()
    # Model arguments
    parser.add_argument('--latent_dim', type=int, default=256)
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
    parser.add_argument('--kl_weight', type=float, default=0.001)
    parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
    parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
    
    args = parser.parse_args()
    network = GaussianVAE(args)


    B = 4
    N = 2048
    pcd = torch.rand((B, N, 3))

    network = network.cuda()
    pcd = pcd.cuda()

    loss = network.get_loss(pcd)

    print(loss)
