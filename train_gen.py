from utils.dataset import ShapeNetCore, SceneVerse
import torch
from torch.utils.data.dataloader import DataLoader
from models.modules import VAEModule
from pytorch_utils.scripts import Trainer
from models.flow import add_spectral_norm

if __name__ == '__main__':
    # Arguments
    trainer = Trainer("Point Diffusion")
    # Model arguments
    trainer.add_argument('--model', type=str, default='flow', choices=['flow', 'gaussian'])
    trainer.add_argument('--latent_dim', type=int, default=256)
    trainer.add_argument('--num_steps', type=int, default=100)
    trainer.add_argument('--beta_1', type=float, default=1e-4)
    trainer.add_argument('--beta_T', type=float, default=0.02)
    trainer.add_argument('--sched_mode', type=str, default='linear')
    trainer.add_argument('--flexibility', type=float, default=0.0)
    trainer.add_argument('--truncate_std', type=float, default=2.0)
    trainer.add_argument('--latent_flow_depth', type=int, default=14)
    trainer.add_argument('--latent_flow_hidden_dim', type=int, default=256)
    trainer.add_argument('--num_samples', type=int, default=4)
    trainer.add_argument('--sample_num_points', type=int, default=2048)
    trainer.add_argument('--kl_weight', type=float, default=0.001)
    trainer.add_argument('--residual', type=eval, default=True, choices=[True, False])
    trainer.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])

    # Datasets and loaders
    trainer.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
    trainer.add_argument('--dataset_name', type=str, default='shapenet', choices=['shapenet', 'sceneverse'])
    trainer.add_argument('--categories', type=list, default=['all'])
    trainer.add_argument('--scale_mode', type=str, default='shape_unit')
    trainer.add_argument('--train_batch_size', type=int, default=128)
    trainer.add_argument('--val_batch_size', type=int, default=64)

    # Optimizer and scheduler
    trainer.add_argument('--learning_rate', type=float, default=2e-3)
    trainer.add_argument('--weight_decay', type=float, default=0)
    trainer.add_argument('--max_grad_norm', type=float, default=10)
    trainer.add_argument('--end_lr', type=float, default=1e-4)
    trainer.add_argument('--sched_start_epoch', type=int, default=200*1000)
    trainer.add_argument('--sched_end_epoch', type=int, default=400*1000)

    args = trainer.setup(train=True, check_val_every_n_epoch=50, gradient_clip_val=0.5)

    torch.set_float32_matmul_precision('high')

    # Datasets and loaders
    
    if args.dataset_name == 'shapenet':
        train_dset = ShapeNetCore(
            path=args.dataset_path,
            cates=args.categories,
            split='train',
            scale_mode=args.scale_mode,
        )
        val_dset = ShapeNetCore(
            path=args.dataset_path,
            cates=args.categories,
            split='val',
            scale_mode=args.scale_mode,
        )
    elif args.dataset_name == 'sceneverse':
        train_dset = SceneVerse(
            path=args.dataset_path,
            split='train',
            scale_mode=args.scale_mode,
            num_points=args.sample_num_points
        )
        val_dset = SceneVerse(
            path=args.dataset_path,
            split='valid',
            scale_mode=args.scale_mode,
            num_points=args.sample_num_points
        )
    else:
        raise ValueError(args.dataset)

    train_loader = DataLoader(train_dset, batch_size=args.train_batch_size, num_workers=args.worker, persistent_workers=True)
    val_loader = DataLoader(val_dset, batch_size=args.val_batch_size, num_workers=args.worker, persistent_workers=True)

    model = VAEModule(args)

    if args.spectral_norm:
        add_spectral_norm(model.model)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)