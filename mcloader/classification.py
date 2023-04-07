import torch
import os
from torch.utils.data import Dataset
from .imagenet import ImageNet


class ClassificationDataset(Dataset):
    """Dataset for classification.
    """

    def __init__(self, split='train', pipeline=None, img_root='/mnt/lustre/share/images'):
        if split == 'train':
            self.data_source = ImageNet(root=os.path.join(img_root, 'train'),
                                        list_file=os.path.join(img_root, 'meta/train.txt'),
                                        memcached=True,
                                        mclient_path='/mnt/lustre/share/memcached_client')
        else:
            self.data_source = ImageNet(root=os.path.join(img_root, 'val'),
                                        list_file=os.path.join(img_root, 'meta/val.txt'),
                                        memcached=True,
                                        mclient_path='/mnt/lustre/share/memcached_client')
        self.pipeline = pipeline

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        img, target = self.data_source.get_sample(idx)
        if self.pipeline is not None:
            img = self.pipeline(img)

        return img, target
