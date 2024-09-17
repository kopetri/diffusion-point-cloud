from utils.dataset import ShapeNetCore
import torch
from torch.utils.data.dataloader import DataLoader
from utils.transform import RandomRotate
from models.modules import AutoEncoderModule
from pytorch_utils.scripts import Trainer

if __name__ == '__main__':
    # Arguments
    trainer = Trainer("Point Diffusion")
    # Model arguments
    trainer.add_argument('--latent_dim', type=int, default=256)
    trainer.add_argument('--num_steps', type=int, default=200)
    trainer.add_argument('--beta_1', type=float, default=1e-4)
    trainer.add_argument('--beta_T', type=float, default=0.05)
    trainer.add_argument('--sched_mode', type=str, default='linear')
    trainer.add_argument('--flexibility', type=float, default=0.0)
    trainer.add_argument('--residual', type=eval, default=True, choices=[True, False])
    trainer.add_argument('--resume', type=str, default=None)

    # Datasets and loaders
    trainer.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
    trainer.add_argument('--categories', type=list, default=['airplane'])
    trainer.add_argument('--scale_mode', type=str, default='shape_unit')
    trainer.add_argument('--train_batch_size', type=int, default=128)
    trainer.add_argument('--val_batch_size', type=int, default=32)
    trainer.add_argument('--rotate', type=eval, default=False, choices=[True, False])

    # Optimizer and scheduler
    trainer.add_argument('--learning_rate', type=float, default=1e-3)
    trainer.add_argument('--weight_decay', type=float, default=0)
    trainer.add_argument('--max_grad_norm', type=float, default=10)
    trainer.add_argument('--end_lr', type=float, default=1e-4)
    trainer.add_argument('--sched_start_epoch', type=int, default=150*1000)
    trainer.add_argument('--sched_end_epoch', type=int, default=300*1000)

    args = trainer.setup()
    
    torch.set_float32_matmul_precision('high')


    # Datasets and loaders
    transform = None
    if args.rotate:
        transform = RandomRotate(180, ['pointcloud'], axis=1)
        
    train_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.categories,
        split='train',
        scale_mode=args.scale_mode,
        transform=transform,
    )
    val_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.categories,
        split='val',
        scale_mode=args.scale_mode,
        transform=transform,
    )

    train_loader = DataLoader(train_dset, batch_size=args.train_batch_size, num_workers=args.worker, persistent_workers=True)
    val_loader = DataLoader(val_dset, batch_size=args.val_batch_size, num_workers=args.worker, persistent_workers=True)

    # Model
    if args.resume is not None:
        model = AutoEncoderModule.load_from_checkpoint(args.resume)
    else:
        model = AutoEncoderModule(args)

    # train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)