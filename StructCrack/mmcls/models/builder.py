"""轻量 Registry 实现，仅保留当前工程需要的功能。"""


class Registry:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.module_dict = {}

    def register_module(self, module=None, name=None):
        def _register(cls):
            key = name or cls.__name__
            self.module_dict[key] = cls
            return cls
        if module is not None:
            return _register(module)
        return _register

    def build(self, cfg):
        if isinstance(cfg, dict):
            cfg = dict(cfg)
            module_type = cfg.pop('type')
            if module_type in self.module_dict:
                return self.module_dict[module_type](**cfg)
            if self.parent is not None:
                return self.parent.build(dict(type=module_type, **cfg))
            raise KeyError(f'{module_type} is not registered in {self.name}')
        raise TypeError(f'cfg must be dict, got {type(cfg)}')


MODELS = Registry('models')
BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
CLASSIFIERS = MODELS
ATTENTION = Registry('attention')


def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_neck(cfg):
    return NECKS.build(cfg)


def build_head(cfg):
    return HEADS.build(cfg)


def build_loss(cfg):
    return LOSSES.build(cfg)


def build_classifier(cfg):
    return CLASSIFIERS.build(cfg)
