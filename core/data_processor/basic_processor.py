import os
import logging
from core.conf.global_config_manager import GlobalConfigManager
from transformers import AutoTokenizer, PreTrainedTokenizer


class BasicProcessor:
    def __init__(self,
                 lm_model_name=None):
        self.lm_model_name = lm_model_name
        self.auto_tokenizer = {}

    def get_auto_tokenizer(self,  model_name: str = None) -> PreTrainedTokenizer:
        if model_name is None:
            model_name = self.lm_model_name
        if model_name not in self.auto_tokenizer:
            logging.info("init {} auto tokenizer".format(model_name))
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=GlobalConfigManager.get_transformers_cache_path())
            self.auto_tokenizer[model_name] = tokenizer
        return self.auto_tokenizer[model_name]

    def delete_all_tokenizer(self):
        for k in self.auto_tokenizer:
            self.auto_tokenizer[k] = None
