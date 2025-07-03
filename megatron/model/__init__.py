# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# type:ignore
# noqa: E401,E402,F401

import torch
from deepspeed.accelerator.real_accelerator import get_accelerator

accelerator = get_accelerator()

if accelerator is not None and accelerator.device_name() == "xpu":
    import intel_extension_for_pytorch  # noqa: F401  # type: ignore

if accelerator is not None and accelerator.device_name() == "cuda":
    from .fused_layer_norm import MixedFusedLayerNorm as LayerNorm
    from .rmsnorm import RMSNorm
    # try:
    #     from apex.normalization import FusedLayerNorm as LayerNorm  # type:ignore
    #     HAS_APEX = True
    # except Exception:
    #     HAS_APEX = False
else:
    if hasattr(torch.xpu, "IpexRmsNorm"):
        from .fused_rmsnorm import RMSNorm
    else:
        from .rmsnorm import RMSNorm  # noqa:E401,E402,F401
    from torch.nn import LayerNorm  # noqa:E401,E402,F401

from .distributed import DistributedDataParallel  # noqa:E401,E402,F401
from .bert_model import BertModel  # noqa:E401,E402,F401
from .gpt_model import GPTModel, GPTModelPipe  # noqa:E401,E402,F401
from .t5_model import T5Model  # noqa:E401,E402,F401
from .language_model import get_language_model  # noqa:E401,E402,F401
from .module import Float16Module  # noqa:E401,E402,F401
