import logging
from core.data_preparer.basic_preparer import BasicPreparer
from datasets import load_dataset
from core.conf.global_config_manager import GlobalConfigManager
from task_emb_2023.conf.train_conf import TrainConfig
from torch.utils.data import TensorDataset, DataLoader
import torch
from tqdm import tqdm
from core.utils.util_structure import UtilStructure


class QaPreparer(BasicPreparer):
    def __init__(self,
                 config: TrainConfig,
                 ):
        super().__init__(lm_model_name=config.lm_model_name)
        self.config = config
        self.squad_read_py_folder_path = GlobalConfigManager.get_current_path() / 'task_emb_2023' / 'data_process' / 'squad_format_datasets'
        self.dataset_names_all = ['squad', 'squad_v2', 'comqa', 'cq', 'drop', 'duorcp', 'duorcs', 'hotpotqa', 'newsqa', 'wikihop']

    def get_dataset_handle(self, dataset_name):
        if dataset_name in ['squad', 'squad_v2']:
            return dataset_name
        else:
            return str(self.squad_read_py_folder_path / (dataset_name + '.py'))

    def preprocess_function(self, examples):
        tokenizer = self.get_auto_tokenizer(self.config.lm_model_name)
        if self.config.lm_model_name.startswith("gpt"):
            tokenizer.add_special_tokens({'pad_token': "*"})
            logging.warning(f'get pad token {tokenizer.pad_token} and pad token id {tokenizer.pad_token_id}')
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=self.config.max_len,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
            return_token_type_ids=True,
            return_attention_mask=True,
        )

        offset_mapping = inputs["offset_mapping"]
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            if len(answer['answer_start']) == 0:
                start_positions.append(0)
                end_positions.append(0)
            else:
                start_char = answer["answer_start"][0]
                end_char = answer["answer_start"][0] + len(answer["text"][0])
                sequence_ids = inputs.sequence_ids(i)

                # Find the start and end of the context
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while sequence_ids[idx] == 1:
                    idx += 1
                    if self.config.lm_model_name.startswith('gpt2') and idx >= len(sequence_ids):
                        break
                context_end = idx - 1

                # If the answer is not fully inside the context, label it (0, 0)
                if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                    start_positions.append(0)
                    end_positions.append(0)
                else:
                    # Otherwise it's the start and end token positions
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)

                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions

        example_ids = []
        post_offset_mapping = []

        for i in range(len(inputs["input_ids"])):
            example_ids.append(examples[i]["id"])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            post_offset_mapping.append([
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ])

        inputs["example_id"] = example_ids
        inputs["post_offset_mapping"] = post_offset_mapping
        return inputs

    def read_data(self, dataset_name):
        dataset = load_dataset(self.get_dataset_handle(dataset_name))
        train = dataset["train"]
        dev = dataset['validation']

        train_input = self.preprocess_function(train)
        dev_input = self.preprocess_function(dev)
        return train, dev, train_input, dev_input

    def get_loader(self, dataset_name):
        train_dataset, dev_dataset, train_input, dev_input = self.read_data(dataset_name)
        datasets = []
        inputs = []
        dataloaders = []
        for split_i, split in enumerate([train_input, dev_input]):
            index = torch.arange(0, len(split["input_ids"]), dtype=torch.long)
            input_ids = torch.tensor(split["input_ids"], dtype=torch.long)
            token_type_ids = torch.tensor(split["token_type_ids"], dtype=torch.long)
            attention_mask = torch.tensor(split["attention_mask"], dtype=torch.long)
            start_positions = torch.tensor(split["start_positions"], dtype=torch.long)
            end_positions = torch.tensor(split["end_positions"], dtype=torch.long)
            # print(index.shape, input_ids.shape, token_type_ids.shape, attention_mask.shape,start_positions.shape,end_positions.shape)
            shuffle = split_i == 0 and self.config.data_loader_shuffle
            dataset = TensorDataset(index, input_ids, token_type_ids, attention_mask, start_positions, end_positions)
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)
            datasets.append([train_dataset, dev_dataset][split_i])
            inputs.append(split)
            dataloaders.append(dataloader)
        assert len(datasets) == len(inputs)
        return [(datasets[i], inputs[i]) for i in range(len(inputs))], dataloaders
