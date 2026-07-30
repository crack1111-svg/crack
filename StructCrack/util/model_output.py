"""统一处理模型输出，兼容 tensor / dict / tuple。"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def get_main_logits(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        return output['logits']
    if isinstance(output, (list, tuple)):
        return output[0]
    return output


def get_aux_logits(output: Any) -> List[torch.Tensor]:
    if isinstance(output, dict):
        return output.get('aux_logits', [])
    return []


def get_boundary_logits(output: Any):
    if isinstance(output, dict):
        return output.get('boundary_logits', None)
    return None
