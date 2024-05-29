import logging
import torch
import torch.utils.data
from core.data_preparer.basic_preparer import BasicPreparer
from task_emb_2023.conf.train_conf import TrainConfig
from core.data_processor.text_classification_processor import TextClassificationProcessor
from core.data_example.text_classification_example import TextClassificationSchema
from torch.utils.data import TensorDataset, DataLoader
from core.utils.util_structure import UtilStructure
import random


class CrPreparer(BasicPreparer):
    def __init__(self,
                 config: TrainConfig,
                 ):
        super().__init__(lm_model_name=config.lm_model_name)
        self.config = config
        self.glue_processor = TextClassificationProcessor()
        self.raw_data = self.glue_processor.read_one_data(config.to_train_dataset)
        self.schema = TextClassificationSchema.DATASET_SCHEMA[self.config.to_train_dataset]
        self.split_ratio = 0.8
        self.split_seed = 42

    def get_loader(self):
        """
        input_ids
        token_type_ids
        attention_mask
        label
        """
        tokenizer = self.get_auto_tokenizer(self.config.lm_model_name)
        if self.config.lm_model_name.startswith("gpt"):
            tokenizer.add_special_tokens({'pad_token': "*"})
            logging.warning(f'get pad token {tokenizer.pad_token} and pad token id {tokenizer.pad_token_id}')
        inp_schema = self.schema['in']
        inp_length = len(inp_schema)
        assert 1 <= inp_length <= 2
        assert self.schema['type'] in ['cls', 'reg']

        if self.config.to_train_dataset in ['anli', 'wanli', 'imdb', 'rotten_tomatoes']:
            all_sets = [self.raw_data['data'][0], self.raw_data['data'][2]]
        else:
            all_sets = self.raw_data['data'][:2]
        assert all_sets[0] is not None
        if all_sets[1] is None:
            logging.warning(f'all_sets[1] is None when processing {self.config.to_train_dataset}, splitting {len(all_sets[0])} with ratio {self.split_ratio} ...')
            train_set = all_sets[0]
            train_len = int(len(train_set) * self.split_ratio)
            rand = random.Random()
            rand.seed(self.split_seed)
            rand.shuffle(train_set)
            dev_set = train_set[train_len:]
            train_set = train_set[:train_len]
            all_sets[0], all_sets[1] = train_set, dev_set
            logging.info(f'split into {len(train_set)} and {len(dev_set)}')

        datasets = []
        dataloaders = []
        for split_i, split in enumerate(all_sets):
            texts = []
            texts_pair = []
            labels = []
            miss_label_num = 0
            for example in split:
                if example['output']['label_num'] is not None:
                    labels.append(example['output']['label_num'])
                    texts.append(example['input'][inp_schema[0]])
                    if inp_length == 2:
                        texts_pair.append(example['input'][inp_schema[1]])
                else:
                    miss_label_num += 1

            if inp_length == 1:
                tokens = tokenizer(text=texts,
                                   padding='longest',
                                   truncation='longest_first',
                                   max_length=self.config.max_len,
                                   return_tensors='pt',
                                   return_token_type_ids=True,
                                   return_attention_mask=True,
                                   )
            elif inp_length == 2:
                tokens = tokenizer(text=texts,
                                   text_pair=texts_pair,
                                   padding='longest',
                                   truncation='longest_first',
                                   max_length=self.config.max_len,
                                   return_tensors='pt',
                                   return_token_type_ids=True,
                                   return_attention_mask=True,
                                   )
            if self.schema['type'] == 'cls':
                labels = torch.tensor(labels, dtype=torch.long)
            elif self.schema['type'] == 'reg':
                labels = torch.tensor(labels, dtype=torch.float32)
            if miss_label_num > 0:
                ratio = miss_label_num/(len(labels) + miss_label_num)
                logging.warning(f"split_i {split_i} miss label {miss_label_num} with ratio {ratio}")
                if ratio > 0.1:
                    raise Exception(f"miss ratio {ratio}")

            shuffle = split_i == 0 and self.config.data_loader_shuffle
            dataset = TensorDataset(tokens.input_ids, tokens.token_type_ids, tokens.attention_mask, labels)
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)
            datasets.append(split)
            dataloaders.append(dataloader)
        return datasets, dataloaders
