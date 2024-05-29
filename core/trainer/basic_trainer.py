import torch
import torch.utils.data
from core.model.basic_model import BasicModel
from core.conf.global_config_manager import GlobalConfigManager
from core.optimizer.basic_optimizer import BasicOptimizer
import json
import os
from core.utils.util_data import UtilData
import logging
from torch.utils.data import DataLoader
from typing import List, Callable
from core.conf.basic_conf import BasicConfig


class BasicTrainer:
    @staticmethod
    def write_json_file(file_name, data):
        with open(file_name, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

    @staticmethod
    def result_folder_init(folder_name):
        path = GlobalConfigManager.get_result_save_path() / folder_name
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def epoch_format(epoch, length):
        epoch_formatted = str(epoch)
        epoch_formatted = '0' * (length - len(epoch_formatted)) + epoch_formatted
        return epoch_formatted

    @staticmethod
    def find_last_epoch_result(folder_path, epoch_positions=(-9, -5)):
        file_names = os.listdir(folder_path)
        max_epoch = -1
        max_file_name = ''
        for file_name in file_names:
            epoch = file_name[epoch_positions[0]:epoch_positions[1]]
            if epoch in ['fig', 'nfig']:
                continue
            epoch = int(epoch)
            if epoch > max_epoch:
                max_epoch = epoch
                max_file_name = file_name
        path = folder_path / max_file_name if max_file_name != '' else None
        return max_file_name, path, max_epoch

    def __init__(self,
                 config: BasicConfig,
                 model: BasicModel,
                 optimizer: BasicOptimizer,
                 train_loader: torch.utils.data.DataLoader,
                 dev_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 ):
        self.config = config
        self.device = config.device
        self.re_set_loader = False
        self.skip_dev = False
        self.skip_save_train_score = False
        self.model: BasicModel = model
        self.optimizer: BasicOptimizer = optimizer
        self.train_loader: torch.utils.data.DataLoader = train_loader
        self.dev_loader: torch.utils.data.DataLoader = dev_loader
        self.test_loader: torch.utils.data.DataLoader = test_loader
        num_training_steps = (len(train_loader) * config.max_epochs) // config.gradient_accumulation_steps + 1
        self.optimizer.prepare_for_train(num_training_steps=num_training_steps,
                                         gradient_accumulation_steps=config.gradient_accumulation_steps)
        if config.model_load_name is not None:
            self.optimizer.load_model(GlobalConfigManager.get_model_save_path() / config.model_load_name)
        self.result_folder_path = self.result_folder_init(config.model_save_name)
        if config.use_data_parallel:
            self.model = torch.nn.DataParallel(self.model)
            logging.info('DataParallel GPU available: {} with {} and with device_ids {}'.format(torch.cuda.device_count(), torch.cuda.current_device(), list(range(config.data_parallel_num))))
        if torch.__version__ >= "2" and config.use_compile:
            self.model = torch.compile(self.model)

    def get_default_init_train_record(self):
        epoch = 0
        best_score = -10000
        patient = self.config.early_stop_patient
        return epoch, best_score, patient, None

    def read_last_epoch_train_record(self, result_folder_path):
        last_file_name, last_file_path, last_epoch = self.find_last_epoch_result(result_folder_path)
        if last_file_path is None:
            epoch, best_score, patient, last_score_result = self.get_default_init_train_record()
        else:
            last_score_result = UtilData.read_raw_json_file(last_file_path)
            epoch = last_score_result['epoch']
            best_score = last_score_result['best_score']
            patient = last_score_result['patient']
        return epoch, best_score, patient, last_score_result

    def save_config(self, result_folder_path):
        config_save_file_name = (self.config.model_save_name + '_config.json')
        config_dict = self.config.__dict__
        config_dict = {k: v if isinstance(v, (int, float, list, bool, dict, tuple, str)) else str(v) for k, v in config_dict.items() if '__' not in k}
        self.write_json_file(result_folder_path / config_save_file_name, config_dict)
        logging.info('score result save to {}'.format(result_folder_path / config_save_file_name))

    def basic_train_template(self,
                             train_batch_fn: Callable,
                             train_args: dict,
                             eval_batch_fn: Callable,
                             eval_args: dict,
                             test_batch_fn: Callable = None,
                             test_args: dict = None,
                             train_loader: DataLoader = None,
                             dev_loader: DataLoader = None,
                             test_loader: DataLoader = None,
                             ):
        test_batch_fn = eval_batch_fn if test_batch_fn is None else test_batch_fn
        test_args = eval_args if test_args is None else test_args
        train_loader = self.train_loader if train_loader is None else train_loader
        dev_loader = self.dev_loader if dev_loader is None else dev_loader
        test_loader = self.test_loader if test_loader is None else test_loader
        if self.config.continue_last_train:
            current_epoch, best_score, patient, best_score_results = self.read_last_epoch_train_record(self.result_folder_path)
        else:
            current_epoch, best_score, patient, best_score_results = self.get_default_init_train_record()
        self.save_config(self.result_folder_path)
        for epoch in range(current_epoch, self.config.max_epochs):
            if self.re_set_loader:
                train_loader = self.train_loader
                dev_loader = self.dev_loader
                test_loader = self.test_loader
                self.re_set_loader = False
            train_score_results = train_batch_fn(dataloader=train_loader, epoch=epoch, **train_args)
            if self.skip_save_train_score:
                train_score_results = None
            self.optimizer.save_model(GlobalConfigManager.get_model_save_path() / (self.config.model_save_name + '_temp.pth'))
            if self.skip_dev:
                dev_score_results = None
            else:
                logging.info('Eval Epoch = {}, dev:'.format(epoch))
                dev_score_results = eval_batch_fn(dataloader=dev_loader, epoch=epoch, **eval_args)
            logging.info('Eval Epoch = {}, test:'.format(epoch))
            test_score_results = test_batch_fn(dataloader=test_loader, epoch=epoch, **test_args)
            final_score_results = {'train': train_score_results,
                                   'dev': dev_score_results,
                                   'test': test_score_results,
                                   "epoch": epoch,
                                   }
            current_score = self.config.stop_score_select(final_score_results)
            if current_score >= best_score:
                self.optimizer.save_model(GlobalConfigManager.get_model_save_path() / (self.config.model_save_name + '_best.pth'))
                best_score = current_score
                best_score_results = final_score_results
                patient = self.config.early_stop_patient
            else:
                patient -= 1
                logging.info('Early Stop Step! score improvement stopped at Epoch {} with patient {}, best score now is {}'.format(
                    epoch, patient, best_score))
            final_score_results.update({'best_score': best_score,
                                        'patient': patient,
                                        })
            epoch_formatted = self.epoch_format(epoch, 4)
            score_results_file_name = self.config.model_save_name + '_' + epoch_formatted + '.json'
            self.write_json_file(self.result_folder_path / score_results_file_name, final_score_results)
            logging.info('score result save to {}'.format(self.result_folder_path / score_results_file_name))
            if self.config.early_stop and patient <= 0:
                break
        logging.info("Train over! best_epoch:{}".format(best_score_results['epoch']))
        # logging.info("Train over! best_score:\n{}".format(json.dumps(best_score_results, ensure_ascii=False, indent=4)))
