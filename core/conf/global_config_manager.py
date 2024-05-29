import logging
import os
import configparser
from pathlib import Path
from dataclasses import dataclass, fields


class GlobalConfigManager:
    """ init the config and current path """
    current_path = Path(os.path.split(os.path.realpath(__file__))[0] + '/../../')  # the path of the code .py file -> the path of the whole project
    logging.info("Current Path: {}".format(current_path))
    config = configparser.ConfigParser()
    config.read(current_path / "core/conf/global.conf")

    @classmethod
    def get_config(cls):
        return cls.config

    @classmethod
    def if_not_exist_then_creat(cls, path):
        if not os.path.exists(path):
            logging.info("Path not exist and creating...: {}, ".format(path))
            os.makedirs(path)

    @classmethod
    def get_current_path(cls):
        return cls.current_path

    @classmethod
    def get_dataset_path(cls, dataset_name):
        path = GlobalConfigManager.current_path / cls.config.get('DATASET', 'main_dataset_path') / dataset_name
        return path

    @classmethod
    def get_temp_dataset_path(cls, dataset_name):
        path = cls.current_path / cls.config.get('DATASET', 'main_dataset_path') / 'Temp' / dataset_name
        return path

    @classmethod
    def get_transformers_cache_path(cls):
        path = cls.current_path / cls.config.get('MODEL', 'transformer_cache_path')
        cls.if_not_exist_then_creat(path)
        return path

    @classmethod
    def get_temp_cache_path(cls):
        path = cls.current_path / cls.config.get('MODEL', 'temp_cache_path')
        return path

    @classmethod
    def get_model_parameter(cls, para_name):
        para = cls.config.get('MODEL', para_name)
        return para

    @classmethod
    def get_model_save_path(cls):
        path = cls.current_path / cls.config.get('MODEL', 'model_save_path')
        cls.if_not_exist_then_creat(path)
        return path

    @classmethod
    def get_metric_cache_path(cls):
        path = cls.current_path / cls.config.get('MODEL', 'metric_save_path')
        cls.if_not_exist_then_creat(path)
        return path

    @classmethod
    def get_result_save_path(cls):
        path = cls.current_path / cls.config.get('MODEL', 'result_save_path')
        cls.if_not_exist_then_creat(path)
        return path

    @classmethod
    def get_out_save_path(cls):
        path = cls.current_path / cls.config.get('MODEL', 'out_save_path')
        cls.if_not_exist_then_creat(path)
        return path

    @classmethod
    def get_word_embedding_cache_path(cls, word_emb_name):
        path = cls.current_path / cls.config.get('MODEL', 'word_embedding_path') / word_emb_name
        cls.if_not_exist_then_creat(path)
        return path

    @classmethod
    def get_ready_dataset_path(cls):
        path = cls.current_path / cls.config.get('DATASET', 'ready_dataset_path')
        return path

    @classmethod
    def get_out_dataset_path(cls):
        path = cls.current_path / cls.config.get('DATASET', 'out_dataset_path')
        return path

    @classmethod
    def get_process_dataset_path(cls):
        path = cls.current_path / cls.config.get('DATASET', 'process_dataset_path')
        return path

    @classmethod
    def config_attribute_transfer(cls, conf_from, conf_to, assert_from_include_to=False):
        res = {}
        for field in fields(conf_from):
            res[field.name] = getattr(conf_from, field.name)

        for field in fields(conf_to):
            if assert_from_include_to:
                assert field.name in res
            if field.name in res:
                setattr(conf_to, field.name, res[field.name])


if __name__ == "__main__":
    print(GlobalConfigManager.current_path)
