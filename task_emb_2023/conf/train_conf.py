from dataclasses import dataclass
from core.conf.basic_conf import BasicConfig


@dataclass
class TrainConfig(BasicConfig):
    task_type: str = None
    pretrain_dataset: str = None
    to_train_dataset: str = None
    cr_to_train_schema: dict = None
    load_path: str = None
    model_save_path: str = None
    score_save_path: str = None
    num_labels: int = None
    qa_n_best_size: int = 20
    qa_max_answer_length: int = 30
