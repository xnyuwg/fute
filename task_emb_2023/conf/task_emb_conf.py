from dataclasses import dataclass
from core.conf.basic_conf import BasicConfig


@dataclass
class TaskEmbConfig(BasicConfig):
    task_type: str = None  # cr qa / llm: nli mcqa
    input_type: str = None  # model data
    task_emb_type: str = None  # fim prefix
    to_learn_model_dataset: str = None
    to_learn_model_lm_name: str = None
    save_path: str = None
    load_path: str = None
    model_save_path: str = None
    cache_path: str = './cache/'
    task_emb_model_lm_name: str = None
    prefix_len: int = None
    early_stop_loss_window: int = None
    max_data_for_task_emb_train: int = None
    model_te_data_shuffle_seed: int = None
    num_labels: int = None
    is_baseline: bool = None
    qa_n_best_size: int = 20
    qa_max_answer_length: int = 30
    fim_pow: float = 2
    llm_mode: bool = None
