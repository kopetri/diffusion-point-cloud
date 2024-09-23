import os
import random
from copy import copy
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import time

def make_split_files(path):
    def __write_txt_(elements, output):
        with open(output, "w") as txtfile:
            for e in elements:
                txtfile.write(e.relative_to(path).as_posix()+"\n")
    path = Path(path)
    print("Generate split files...")
    
    pths = list(path.glob("**/pcd_with_global_alignment/*.pth"))
    assert len(pths) > 0, len(pths)
    print(len(pths))
    random.shuffle(pths)
    N = len(pths)
    N_train = int(0.8 * N)
    N_valid = int(0.1 * N)
    train_pths = pths[0:N_train]
    valid_pths = pths[N_train:N_train+N_valid]
    test_pths  = pths[N_train+N_valid:]
    
    train_pths = sorted(train_pths)
    valid_pths = sorted(valid_pths)
    test_pths = sorted(test_pths)
    
    __write_txt_(train_pths, path/"train.txt")
    __write_txt_(valid_pths, path/"valid.txt")
    __write_txt_(test_pths, path/"test.txt")

def make_hdf5_files(path):
    
    print("Generate split files...")
    
    pths = list(Path(path).glob("**/pcd_with_global_alignment/*.pth"))
    assert len(pths) > 0, len(pths)
    print(len(pths))
    random.shuffle(pths)
    N = len(pths)
    N_train = int(0.8 * N)
    N_valid = int(0.1 * N)
    train_pths = pths[0:N_train]
    valid_pths = pths[N_train:N_train+N_valid]
    test_pths  = pths[N_train+N_valid:]
    
    train_pths = sorted(train_pths)
    valid_pths = sorted(valid_pths)
    test_pths = sorted(test_pths)
    
    for split, pths_ in zip(["valid", "train", "test"], [valid_pths, train_pths, test_pths]):
        hdf5_file = Path(path, f"{split}.hdf5")
        with h5py.File(hdf5_file.as_posix(), 'w') as hdf:
            for pth in tqdm(pths_, desc=f"Loading {split}"):
                assert pth.exists(), pth
                data = torch.load(pth)
                group_name = pth.stem
                try:
                    pos, color = data[0:2]
                    positions, colors = (pos.astype(np.float32), color.astype(np.uint8))
                except:
                    print(data)
                    print(len(data))
                    import sys; sys.exit()
                group = hdf.create_group(group_name)
                group.create_dataset('positions', data=positions, dtype='float32')
                group.create_dataset('colors', data=colors, dtype='uint8')
                

synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


class ShapeNetCore(Dataset):

    GRAVITATIONAL_AXIS = 1
    
    def __init__(self, path, cates, split, scale_mode, transform=None):
        super().__init__()
        assert isinstance(cates, list), '`cates` must be a list of cate names.'
        assert split in ('train', 'val', 'test')
        assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34')
        self.path = path
        if 'all' in cates:
            cates = cate_to_synsetid.keys()
        self.cate_synsetids = [cate_to_synsetid[s] for s in cates]
        self.cate_synsetids.sort()
        self.split = split
        self.scale_mode = scale_mode
        self.transform = transform

        self.pointclouds = []
        self.stats = None

        self.get_statistics()
        self.load()

    def get_statistics(self):

        basename = os.path.basename(self.path)
        dsetname = basename[:basename.rfind('.')]
        stats_dir = os.path.join(os.path.dirname(self.path), dsetname + '_stats')
        os.makedirs(stats_dir, exist_ok=True)

        if len(self.cate_synsetids) == len(cate_to_synsetid):
            stats_save_path = os.path.join(stats_dir, 'stats_all.pt')
        else:
            stats_save_path = os.path.join(stats_dir, 'stats_' + '_'.join(self.cate_synsetids) + '.pt')
        if os.path.exists(stats_save_path):
            self.stats = torch.load(stats_save_path)
            return self.stats

        with h5py.File(self.path, 'r') as f:
            pointclouds = []
            for synsetid in self.cate_synsetids:
                for split in ('train', 'val', 'test'):
                    pointclouds.append(torch.from_numpy(f[synsetid][split][...]))

        all_points = torch.cat(pointclouds, dim=0) # (B, N, 3)
        B, N, _ = all_points.size()
        mean = all_points.view(B*N, -1).mean(dim=0) # (1, 3)
        std = all_points.view(-1).std(dim=0)        # (1, )

        self.stats = {'mean': mean, 'std': std}
        torch.save(self.stats, stats_save_path)
        return self.stats

    def load(self):

        def _enumerate_pointclouds(f):
            for synsetid in self.cate_synsetids:
                cate_name = synsetid_to_cate[synsetid]
                for j, pc in enumerate(f[synsetid][self.split]):
                    yield torch.from_numpy(pc), j, cate_name
        
        with h5py.File(self.path, mode='r') as f:
            for pc, pc_id, cate_name in _enumerate_pointclouds(f):

                if self.scale_mode == 'global_unit':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = self.stats['std'].reshape(1, 1)
                elif self.scale_mode == 'shape_unit':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1)
                elif self.scale_mode == 'shape_half':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.5)
                elif self.scale_mode == 'shape_34':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.75)
                elif self.scale_mode == 'shape_bbox':
                    pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
                    pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
                    shift = ((pc_min + pc_max) / 2).view(1, 3)
                    scale = (pc_max - pc_min).max().reshape(1, 1) / 2
                else:
                    shift = torch.zeros([1, 3])
                    scale = torch.ones([1, 1])

                pc = (pc - shift) / scale

                self.pointclouds.append({
                    'pointcloud': pc,
                    'cate': cate_name,
                    'id': pc_id,
                    'shift': shift,
                    'scale': scale
                })

        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.Random(2020).shuffle(self.pointclouds)

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)
        return data
    
    
class SceneVerse(Dataset):

    GRAVITATIONAL_AXIS = 1
    
    def __init__(self, path, split, scale_mode, transform=None, num_points=2048):
        super().__init__()
        assert split in ('train', 'valid', 'test')
        assert scale_mode is None or scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34')
        self.path = Path(path)
        self.split = split
        self.scale_mode = scale_mode
        self.transform = transform
        self.num_points = num_points
        self.stats = None
        start = time.time()
        self.pointclouds = []
        self.load()
        #self.pointclouds = self.data[self.split].keys()
        #self.pointclouds = [k for k in self.data.keys()]
        print(f"Found {self.__len__()} datapoints for {self.split} in {time.time() - start:0.2f} seconds")
        
    def load(self):

        def _enumerate_pointclouds(f):
            for group_name in tqdm(f.keys()):
                group = f[group_name]
                positions = group["positions"][:]
                colors = group["colors"][:]
                yield torch.from_numpy(positions).float(), torch.from_numpy(colors), group_name
        
        with h5py.File(self.path/f"{self.split}.hdf5", mode='r') as f:
            for pc, colors, pc_id in _enumerate_pointclouds(f):

                if self.scale_mode == 'global_unit':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = self.stats['std'].reshape(1, 1)
                elif self.scale_mode == 'shape_unit':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1)
                elif self.scale_mode == 'shape_half':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.5)
                elif self.scale_mode == 'shape_34':
                    shift = pc.mean(dim=0).reshape(1, 3)
                    scale = pc.flatten().std().reshape(1, 1) / (0.75)
                elif self.scale_mode == 'shape_bbox':
                    pc_max, _ = pc.max(dim=0, keepdim=True) # (1, 3)
                    pc_min, _ = pc.min(dim=0, keepdim=True) # (1, 3)
                    shift = ((pc_min + pc_max) / 2).view(1, 3)
                    scale = (pc_max - pc_min).max().reshape(1, 1) / 2
                else:
                    shift = torch.zeros([1, 3])
                    scale = torch.ones([1, 1])

                pc = (pc - shift) / scale

                self.pointclouds.append({
                    'pointcloud': pc,
                    'id': pc_id,
                    'shift': shift,
                    'scale': scale
                })

        # Deterministically shuffle the dataset
        self.pointclouds.sort(key=lambda data: data['id'], reverse=False)
        random.Random(2020).shuffle(self.pointclouds)
    
    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {k:v.clone() if isinstance(v, torch.Tensor) else copy(v) for k, v in self.pointclouds[idx].items()}
        if self.transform is not None:
            data = self.transform(data)
        return data

if __name__ == "__main__":
    #make_split_files("data/sceneverse")
    dataset = SceneVerse("data/sceneverse", "train", "shape_unit")
    for dp in dataset:
        print(dp)
        break
    dataset = SceneVerse("data/sceneverse", "valid", "shape_unit")
    for dp in dataset:
        print(dp)
        break
    dataset = SceneVerse("data/sceneverse", "test", "shape_unit")
    for dp in dataset:
        print(dp)
        break
    