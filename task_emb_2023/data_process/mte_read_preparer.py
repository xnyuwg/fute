import logging
import torch
import torch.utils.data
from core.data_preparer.basic_preparer import BasicPreparer
from task_emb_2023.conf.task_emb_conf import TaskEmbConfig
from task_emb_2023.conf.train_conf import TrainConfig
from torch.utils.data import TensorDataset, DataLoader
from core.utils.util_data import UtilData
from core.conf.global_config_manager import GlobalConfigManager
import numpy
from core.utils.util_math import UtilMath


class MteReadPreparer(BasicPreparer):
    def __init__(self,
                 lm_model_name,
                 ):
        super().__init__(lm_model_name=lm_model_name)

    def read_data(self, config: TaskEmbConfig):
        path = config.load_path
        raw_data = UtilData.read_raw_jsonl_file(path)
        data = {}
        for rd in raw_data:
            data_index = rd['data_index']
            prompt_info = rd['prompt_info']
            answer_choices = prompt_info['answer_choices']
            data_entry = rd["data_entry"]
            token_info = rd['token_info']
            word_scores = token_info['word_scores']
            verbalizers = token_info['verbalizers']
            assert len(word_scores) == len(verbalizers)
            verb_to_answer = {v: k for k, vs in answer_choices.items() for v in vs}
            label_names = answer_choices.keys()
            final_score_dict = {k: 0 for k in answer_choices}
            for ws, vb in zip(word_scores, verbalizers):
                pred_name = verb_to_answer[vb]
                pred_score = ws
                if pred_score > final_score_dict[pred_name]:
                    final_score_dict[pred_name] = ws
            scores = [final_score_dict[ln] for ln in label_names]
            prob = UtilMath.numpy_softmax(numpy.array(scores)).tolist()
            assert 0.99 < sum(prob) < 1.01
            assert len(label_names) == len(scores) == len(prob)
            if config.task_type == 'nli':
                d = {
                    'premise': data_entry['premise'],
                    'hypothesis': data_entry['hypothesis'],
                }
                assert len(prob) == 3
            elif config.task_type == 'sa':
                d = {
                    'context': data_entry['text'],
                }
                assert len(prob) == 2
            d.update({
                'label_names': label_names,
                'scores': scores,
                'label': prob,
            })
            data[data_index] = d
        data_out = [(k, v) for k, v in data.items()]
        data_out = sorted(data_out, key=lambda x: x[0])
        data_out = [x[1] for x in data_out]
        logging.info(f'read {len(data_out)} data from {path}')
        return data_out

    def get_loader(self, config: TaskEmbConfig, model_config: TrainConfig):
        data = self.read_data(config)
        tokenizer = self.get_auto_tokenizer(config.task_emb_model_lm_name)
        if config.task_emb_model_lm_name.startswith("gpt"):
            tokenizer.add_special_tokens({'pad_token': "*"})
            logging.warning(f'get pad token {tokenizer.pad_token} and pad token id {tokenizer.pad_token_id}')

        texts = []
        texts_pair = []
        labels = []
        for example in data:
            labels.append(example['label'])
            if config.task_type.startswith('sa'):
                texts.append(example['context'])
            elif config.task_type.startswith('nli'):
                texts.append(example['premise'])
                texts_pair.append(example['hypothesis'])

        if config.task_type.startswith('sa'):
            tokens = tokenizer(text=texts,
                               padding='longest',
                               truncation='longest_first',
                               max_length=config.max_len,
                               return_tensors='pt',
                               return_token_type_ids=True,
                               return_attention_mask=True,
                               )
        elif config.task_type.startswith('nli'):
            tokens = tokenizer(text=texts,
                               text_pair=texts_pair,
                               padding='longest',
                               truncation='longest_first',
                               max_length=config.max_len,
                               return_tensors='pt',
                               return_token_type_ids=True,
                               return_attention_mask=True,
                               )
        labels = torch.tensor(labels, dtype=torch.float32)

        dataset = TensorDataset(tokens.input_ids, tokens.token_type_ids, tokens.attention_mask, labels)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
        return dataset, dataloader
