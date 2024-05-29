import logging
from typing import List
import torch
import torch.utils.data
from transformers import BertTokenizer, AutoTokenizer, PreTrainedTokenizer
from core.conf.global_config_manager import GlobalConfigManager


class BasicPreparer:
    @staticmethod
    def padding_to_max_len(arrays: List[list], padding_token):
        max_len = max(len(array) for array in arrays)
        new_arrays = []
        for array in arrays:
            padding_len = max_len - len(array)
            new_array = array + [padding_token] * padding_len
            new_arrays.append(new_array)
        for array in new_arrays:
            assert len(array) == len(new_arrays[0])
        return new_arrays

    @staticmethod
    def fn_to_get_data_loader(batch_size: int, collate_fn, data_loader_shuffle: bool, data_loader_worker_num: int):
        def get_data_loader(dataset: torch.utils.data.Dataset) -> torch.utils.data.DataLoader:
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=data_loader_shuffle, num_workers=data_loader_worker_num)
            return data_loader
        return get_data_loader

    def __init__(self, lm_model_name):
        """ init """
        self.lm_model_name = lm_model_name
        self.__bert_tokenizer = None
        self.__auto_tokenizer = {}

    def get_bert_tokenizer(self, model_name: str = None) -> BertTokenizer:
        """ please use this method to get bert tokenizer """
        if model_name is None:
            model_name = 'bert-base-uncased'
        if self.__bert_tokenizer is None:
            logging.info("init bert_tokenizer")
            self.__bert_tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=GlobalConfigManager.get_transformers_cache_path())
        return self.__bert_tokenizer

    def get_auto_tokenizer(self,  model_name: str = None) -> PreTrainedTokenizer:
        if model_name is None:
            model_name = self.lm_model_name
        if model_name not in self.__auto_tokenizer:
            logging.info("init {} auto tokenizer".format(model_name))
            if model_name == "fnlp/bart-base-chinese":
                tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=GlobalConfigManager.get_transformers_cache_path())
            elif model_name == "hfl/chinese-roberta-wwm-ext":
                tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=GlobalConfigManager.get_transformers_cache_path())
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=GlobalConfigManager.get_transformers_cache_path())
            self.__auto_tokenizer[model_name] = tokenizer
        return self.__auto_tokenizer[model_name]



