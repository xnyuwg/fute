import logging
from core.data_processor.basic_processor import BasicProcessor
from core.conf.global_config_manager import GlobalConfigManager
import random
import os
import sys
import importlib
import time


class CrossfitProcessor(BasicProcessor):
    def __init__(self, max_data_num=None, choose_seed=None):
        super().__init__()
        self.crossfit_folder_path = GlobalConfigManager.get_current_path() / 'core' / 'data_processor' / 'crossfit'
        logging.info(f'CrossFit folder path: {self.crossfit_folder_path}')
        self.max_data_num = max_data_num
        if max_data_num is not None:
            assert choose_seed is not None
        self.choose_seed = choose_seed

    def read_data(self, read_list, input_num):
        total_start_time = time.time()
        logging.info(f'starting reading data for {read_list}, total num: {len(read_list)}')
        sys.path.append(str(self.crossfit_folder_path))
        sys.path.append(str(self.crossfit_folder_path / 'crossfit_data_process'))

        all_data = []
        dataset_index = {}
        current_index = 0
        read_i = 0
        for file_name in read_list:
            start_time = time.time()
            # logging.info('reading {}'.format(file_name))
            file_path = self.crossfit_folder_path / 'crossfit_dataset' / (file_name + '.py')
            data, size_num = self.load_one_data(file_path, input_num)
            dataset_index[file_name] = [current_index, current_index + len(data)]
            current_index += len(data)
            all_data += data
            used_time = (time.time() - start_time) / 60
            read_i += 1
            logging.info('read_done={}, size={}, actual_size={}, progress={}/{} used_time={:.2f}'.format(file_name, size_num, len(data), read_i, len(read_list), used_time))

        sys.path.remove(str(self.crossfit_folder_path))
        sys.path.remove(str(self.crossfit_folder_path / 'crossfit_data_process'))
        total_used_time = (time.time() - total_start_time) / 60
        logging.info('read done all, total used time: {:.2f}'.format(total_used_time))
        return all_data, dataset_index

    def load_one_data(self, file_path, input_num=(0, 1e10)):
        read_object = self.load_class_from_file(file_path)
        train_lines, test_lines = read_object.get_data()
        size_num = len(train_lines)
        data_line = train_lines[0]
        text = data_line[0]
        texts = text.split('[SEP]')
        texts = [x.strip() for x in texts]
        texts = [x for x in texts if len(x) > 0]
        if not (input_num[0] <= len(texts) <= input_num[1]):
            # logging.warning(f'input number {len(texts)} for {file_path}')
            pass

        if self.max_data_num is not None and len(train_lines) > self.max_data_num:
            rand = random.Random()
            rand.seed(self.choose_seed)
            rand.shuffle(train_lines)
            train_lines = train_lines[:self.max_data_num]

        return train_lines, size_num

    def load_class_from_file(self, file_path):
        file_path = os.path.normpath(file_path)
        directory, filename = os.path.split(file_path)
        module_name = filename.replace('.py', '')
        module = importlib.import_module('crossfit_dataset.' + module_name)
        count = 3
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if isinstance(attribute, type):
                if str(attribute.__name__) not in ['FewshotGymDataset', 'FewshotGymClassificationDataset', 'FewshotGymTextToTextDataset']:
                    return attribute()
