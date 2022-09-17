import torch

from datasets.AGNewsDataset import AGNewsDataset
from configs.dataset_config import cfg as dataset_cfg
from configs.train_config import cfg as train_cfg


def get_dataloaders():
    """
    Initializes train, test datasets and gets their dataloaders.
    :return: train and test dataloaders
    """
    train_dataset = AGNewsDataset(cfg=dataset_cfg, dataset_type='train')
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=train_cfg.batch_size, drop_last=True,
                                           pin_memory=True)
    test_dataset = AGNewsDataset(cfg=dataset_cfg, dataset_type='test')
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=train_cfg.batch_size, pin_memory=True)
    return train_dl, test_dl
