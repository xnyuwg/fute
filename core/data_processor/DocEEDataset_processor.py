import logging
from core.data_processor.basic_processor import BasicProcessor
from core.conf.global_config_manager import GlobalConfigManager
from core.utils.util_data import UtilData


class DocEEDatasetProcessor(BasicProcessor):
    def __init__(self):
        super().__init__()
        self.data_path = GlobalConfigManager.get_dataset_path('DocEEDataset')
        logging.info("DocEEDataset Path: {}".format(self.data_path))

        normal_setting_path = self.data_path / 'normal_setting'
        split_path_name = ['train', 'dev', 'test']
        all_normal_json_path = [normal_setting_path / (x + '.json') for x in split_path_name]

        self.all_normal_json = [UtilData.read_raw_jsonl_file(x) for x in all_normal_json_path]