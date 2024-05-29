import logging
from core.data_processor.basic_processor import BasicProcessor
from core.conf.global_config_manager import GlobalConfigManager
from core.utils.util_data import UtilData


class ACE2005Processor(BasicProcessor):
    def __init__(self):
        super().__init__()
        self.data_path = GlobalConfigManager.get_dataset_path('ACE2005')
        logging.info("ACE2005 Path: {}".format(self.data_path))

        self.english_path = self.data_path / 'EnglishACE2005' / 'output'

        self.english_train_path = self.english_path / "train.json"
        self.english_dev_path = self.english_path / "dev.json"
        self.english_test_path = self.english_path / "test.json"
        logging.debug("english_train_path: {}".format(self.english_train_path))
        logging.debug("english_dev_path: {}".format(self.english_dev_path))
        logging.debug("english_test_path: {}".format(self.english_test_path))

        self.english_train_json = UtilData.read_raw_json_file(self.english_train_path)
        self.english_dev_json = UtilData.read_raw_json_file(self.english_dev_path)
        self.english_test_json = UtilData.read_raw_json_file(self.english_test_path)
        self.english_all_json = [self.english_train_json, self.english_dev_json, self.english_test_json]
