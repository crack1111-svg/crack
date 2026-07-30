"""轻量兼容版 mmcls 入口。

原始仓库依赖 mmcv/mmengine；为便于独立运行，这里保留最小接口。
"""

from .version import __version__

__all__ = ['__version__']
