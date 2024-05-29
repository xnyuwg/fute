from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable
import torch


@dataclass
class BasicConfig:
    debug: bool = None
    model_load_name: str = None
    model_save_name: str = None
    batch_size: int = None
    early_stop_patient: int = None
    early_stop: bool = None
    data_loader_shuffle: bool = None
    lm_model_name: str = None
    device: torch.device = None
    max_len: int = None
    learning_rate_slow: float = 1e-5
    learning_rate_fast: float = 1e-4
    gradient_accumulation_steps: int = None
    max_epochs: int = None
    continue_last_train: bool = None
    scheduler_type: str = None
    scheduler_linear_warmup_proportion: float = 0.05
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    dropout_ratio: float = 0.15
    label_smoothing: float = 0.0
    optimizing_no_decay: str = "bias,LayerNorm.bias,LayerNorm.weight"
    stop_score_select: Callable = None
    use_compile: bool = False
    use_amp: bool = False
    use_data_parallel: bool = False
    data_parallel_num: int = None
    use_distributed_data_parallel: bool = False
    distributed_data_parallel_num: int = None
    num_works: int = 0
