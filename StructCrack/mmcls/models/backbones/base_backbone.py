from abc import ABCMeta, abstractmethod

import torch.nn as nn


class BaseBackbone(nn.Module, metaclass=ABCMeta):
    """轻量版 BaseBackbone。"""

    def __init__(self, init_cfg=None):
        super().__init__()
        self.init_cfg = init_cfg

    def init_weights(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass

    def train(self, mode=True):
        super().train(mode)
        return self
