"""数据集抽象基类。"""

from abc import ABC, abstractmethod
import torch.utils.data as data


class BaseDataset(data.Dataset, ABC):
    def __init__(self, args):
        self.args = args
        self.root = args.dataset_path

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError
