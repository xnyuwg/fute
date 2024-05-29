import argparse
import importlib
import logging
import torch
from task_emb_2023.conf.train_conf import TrainConfig
from task_emb_2023.data_process.cr_preparer import CrPreparer
from task_emb_2023.model.cr_model import CrModelV1
from core.data_example.text_classification_example import TextClassificationSchema
from core.optimizer.basic_optimizer import BasicOptimizer
from task_emb_2023.trainer.train_trainer import TrainTrainer
from core.conf.global_config_manager import GlobalConfigManager
importlib.reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')


def get_config(args) -> TrainConfig:
    config = TrainConfig()

    config.task_type = 'cr'
    config.lm_model_name = args.lm_model_name
    config.to_train_dataset = args.to_train_dataset
    config.load_path = args.load_path
    config.model_save_path = args.model_save_path
    config.score_save_path = args.score_save_path
    if not config.score_save_path.endswith('.json'):
        config.score_save_path += '.json'

    GlobalConfigManager.if_not_exist_then_creat(config.model_save_path)

    config.cr_to_train_schema = TextClassificationSchema.DATASET_SCHEMA[config.to_train_dataset]
    if config.cr_to_train_schema['type'] == 'reg':
        config.num_labels = 1
    elif config.cr_to_train_schema['type'] == 'cls':
        config.num_labels = len(config.cr_to_train_schema['label'])

    config.device = torch.device('cuda')
    config.data_loader_shuffle = True
    config.max_epochs = 3
    config.max_len = 128
    config.gradient_accumulation_steps = 1 if args.gradient_accumulation_steps is None else args.gradient_accumulation_steps
    config.batch_size = 32 // config.gradient_accumulation_steps
    learning_rate = 2e-5
    config.learning_rate_slow = learning_rate
    config.learning_rate_fast = learning_rate
    config.scheduler_type = 'linear'
    config.early_stop = False
    return config


def run(args):
    config = get_config(args)
    logging.info(f'start new run_one to_train_dataset={config.to_train_dataset}, load_path={config.load_path}, model_save_path={config.model_save_path}, score_save_path={config.score_save_path}')

    pro = CrPreparer(config)
    datasets, dataloaders = pro.get_loader()

    model = CrModelV1(config=config)
    model.to(config.device)

    optimizer = BasicOptimizer(config=config,
                               model=model,
                               )

    trainer = TrainTrainer(config=config,
                           model=model,
                           optimizer=optimizer,
                           train_loader=dataloaders[0],
                           dev_loader=dataloaders[1],
                           )
    trainer.train()


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--to_train_dataset", type=str, required=True)
    arg_parser.add_argument("--lm_model_name", type=str, required=True)
    arg_parser.add_argument("--model_save_path", type=str, required=True)
    arg_parser.add_argument("--score_save_path", type=str, required=True)
    arg_parser.add_argument("--load_path", type=str, required=False, default=None)
    arg_parser.add_argument("--gradient_accumulation_steps", type=int, required=False, default=None)
    args = arg_parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    run(args)
