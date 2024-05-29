import logging
from basic_processor import BasicProcessor
from core.conf.global_config_manager import GlobalConfigManager
from core.utils.util_data import UtilData


class MAVENProcessor(BasicProcessor):
    def __init__(self):
        """ init the MAVEN path and read the raw data """
        super().__init__()
        self.MAVEN_path = GlobalConfigManager.get_dataset_path('MAVEN')
        logging.info("MAVEN Path: {}".format(self.MAVEN_path))

        self.train_jsonl_path = self.MAVEN_path / "train.jsonl"
        self.valid_jsonl_path = self.MAVEN_path / "valid.jsonl"
        self.test_jsonl_path = self.MAVEN_path / "test.jsonl"
        logging.debug("train_jsonl_path: {}".format(self.train_jsonl_path))
        logging.debug("valid_jsonl_path: {}".format(self.valid_jsonl_path))
        logging.debug("test_jsonl_path: {}".format(self.test_jsonl_path))

        # read data
        logging.info("reading data...")
        self.train_raw_json_list = UtilData.read_raw_jsonl_file(self.train_jsonl_path)
        self.valid_raw_json_list = UtilData.read_raw_jsonl_file(self.valid_jsonl_path)
        self.test_raw_json_list = UtilData.read_raw_jsonl_file(self.test_jsonl_path)


if __name__ == "__main__":
    # only for test
    m = MAVENProcessor()

