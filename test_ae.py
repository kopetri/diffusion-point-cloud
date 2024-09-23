import torch
import numpy as np
from pathlib import Path
from utils.dataset import ShapeNetCore
from torch.utils.data.dataloader import DataLoader
from pytorch_utils.scripts import Trainer
from models.modules import AutoEncoderModule
from evaluation import EMD_CD

if __name__ == '__main__':
    # Arguments
    trainer = Trainer("Evaluation")

    trainer.add_argument('--ckpt', type=str, required=True)
    trainer.add_argument('--categories', type=list, default=['airplane'])

    # Datasets and loaders
    trainer.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
    trainer.add_argument('--batch_size', type=int, default=128)

    args = trainer.setup(train=False)
    
    save_dir = Path(args.ckpt).parents[1]/"predictions"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    torch.set_float32_matmul_precision('high')

    # model
    module = AutoEncoderModule.load_from_checkpoint(args.ckpt)

    test_dset = ShapeNetCore(
        path=args.dataset_path,
        cates=args.categories,
        split='test',
        scale_mode=module.opt.scale_mode
    )
    test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=4, persistent_workers=True)

    result = trainer.predict(module, dataloaders=test_loader)
    
    all_ref = [r[0] for r in result]
    all_recons = [r[1] for r in result]
    
    all_ref = torch.cat(all_ref, dim=0)
    all_recons = torch.cat(all_recons, dim=0)

    print(f'Saving point clouds... to {save_dir}')
    np.save(save_dir/'ref.npy', all_ref.numpy())
    np.save(save_dir/'out.npy', all_recons.numpy())

    print('Start computing metrics...')
    metrics = EMD_CD(all_recons.cuda(), all_ref.cuda(), batch_size=args.batch_size)
    cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
    print('CD:  %.12f' % cd)
    print('EMD: %.12f' % emd)