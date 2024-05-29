import logging
from torch.utils.data import Dataset, DataLoader
import random

from transformers import PreTrainedTokenizer, AutoTokenizer

from core.data_preparer.basic_preparer import BasicPreparer
from task_emb_2023.conf.task_emb_conf import TaskEmbConfig
from task_emb_2023.conf.train_conf import TrainConfig
from core.data_example.text_classification_example import TextClassificationSchema
from core.data_processor.crossfit_processor import CrossfitProcessor


class CrossfitPreparer(BasicPreparer):
    def __init__(self,
                 lm_model_name,
                 data_lines,
                 max_length=512,
                 ):
        super().__init__(lm_model_name=lm_model_name)
        self.data_lines = data_lines
        self.new_data_lines = None
        self.max_length = max_length

    def get_new_data_lines(self, config):
        if self.new_data_lines is not None:
            return self.new_data_lines
        else:
            new_data_lines = []
            for data_line in self.data_lines:
                text = data_line[0]
                texts = text.split('[SEP]')
                texts = [x.strip() for x in texts]
                texts = [x for x in texts if len(x) > 0]
                if 1 <= len(texts):
                    if len(texts) > 2:
                        texts = texts[:2]
                    new_data_lines.append(texts)
            rand = random.Random()
            rand.seed(config.model_te_data_shuffle_seed)
            rand.shuffle(new_data_lines)
            self.new_data_lines = new_data_lines
            self.data_lines = None
            random_check = random.randint(0, len(self.new_data_lines) - 11)
            logging.info(f'sample start from {random_check} of crossfit input: {self.new_data_lines[random_check: random_check + 10]}')
            return self.new_data_lines

    def generate_model_data_mcqa(self, data, config: TaskEmbConfig):
        texts = []
        if len(data) == 3:
            question = data[0]
            context = data[1]
            answers = self.find_answer_according_to_abcd_else_split(data[2])
            texts.append({
                'question': question,
                'context': context,
                'answers': answers,
            })
        elif len(data) == 2:
            question = data[0]
            context = data[1]
            answers = self.find_answer_according_to_abcd_else_split(context)
            texts.append({
                'question': question,
                'context': context,
                'answers': answers,
            })
        elif len(data) == 1:
            text = data[0]
            if "(A)" in text:
                a_index = text.find("(A)")
                assert a_index != -1
                qc = text[:a_index]
                q_len = len(qc) // 3
                question = qc[:q_len]
                context = qc[q_len:]
                answers = text[a_index:]
                answers = self.find_answer_according_to_abcd_else_split(answers)
                texts.append({
                    'question': question,
                    'context': context,
                    'answers': answers,
                })
            else:
                q_len = len(text) // 3
                question = text[:q_len]
                context = text[q_len:]
                answers = self.find_answer_according_to_abcd_else_split(context)
                texts.append({
                    'question': question,
                    'context': context,
                    'answers': answers,
                })
        return texts

    def generate_model_data_nli(self, data_lines, config: TaskEmbConfig):
        tokenizer = self.get_auto_tokenizer(config.task_emb_model_lm_name)
        texts = []
        for data in data_lines:
            if len(data) == 2:
                texts.append({
                    'premise': data[0],
                    'hypothesis': data[1],
                    'text': 2,
                })
            elif len(data) == 1:
                text = data[0]
                if 'premise:' in text.lower():
                    premise_index = text.lower().find('premise:')
                    assert premise_index != -1
                    hypothesis_index = text.find('hypothesis:')
                    if hypothesis_index == -1:
                        hypothesis_index = (premise_index + len('premise') + len(text)) // 2
                    if hypothesis_index > premise_index:
                        premise = text[:hypothesis_index]
                        hypothesis_index = text[hypothesis_index:]
                    else:
                        hypothesis_index = text[:premise_index]
                        premise = text[premise_index:]
                    texts.append({
                        'premise': premise,
                        'hypothesis': hypothesis_index,
                        'text': 1,
                    })
                else:
                    mid = len(text) // 2
                    texts.append({
                        'premise': text[:mid],
                        'hypothesis': text[mid:],
                        'text': 3,
                    })

        for text in texts:
            premise_tokens = tokenizer.encode(text['premise'], add_special_tokens=False)
            hypothesis_tokens = tokenizer.encode(text['hypothesis'], add_special_tokens=False)
            max_length = self.max_length - 2
            if len(premise_tokens) + len(hypothesis_tokens) > max_length:
                min_length = min(len(premise_tokens), len(hypothesis_tokens))
                per_max_length = max(max_length // 2, max_length - min_length)
                premise_tokens = premise_tokens[:per_max_length]
                hypothesis_tokens = hypothesis_tokens[:per_max_length]
            assert len(premise_tokens) + len(hypothesis_tokens) <= max_length
            premise = tokenizer.decode(premise_tokens, skip_special_tokens=True)
            hypothesis = tokenizer.decode(hypothesis_tokens, skip_special_tokens=True)
            text['premise'] = premise
            text['hypothesis'] = hypothesis
        return texts

    def generate_model_data_sa(self, data_lines, config: TaskEmbConfig):
        tokenizer = self.get_auto_tokenizer(config.task_emb_model_lm_name)
        texts = []
        for data in data_lines:
            if len(data) == 2:
                texts.append({
                    'content': data[1],
                    'text': 2,
                })
            elif len(data) == 1:
                texts.append({
                    'content': data[0],
                    'text': 1,
                })
        for text in texts:
            content_tokens = tokenizer.encode(text['content'], add_special_tokens=False)
            max_length = self.max_length - 1
            if len(content_tokens) > max_length:
                content_tokens = content_tokens[:max_length]
            assert len(content_tokens) <= max_length
            content = tokenizer.decode(content_tokens, skip_special_tokens=True)
            text['content'] = content
        return texts

    def generate_model_data_entry(self, config: TaskEmbConfig, max_data_num):
        new_data_lines = self.get_new_data_lines(config)
        logging.info(f'get data lines: {len(new_data_lines)}')
        if config.task_type == 'nli':
            texts = self.generate_model_data_nli(new_data_lines, config)
        elif config.task_type == 'sa':
            texts = self.generate_model_data_sa(new_data_lines, config)
        logging.info(f'get data texts: {len(texts)}')
        if len(texts) > max_data_num:
            texts = sorted(texts, key=lambda x: x['text'])
            texts = texts[:max_data_num]
            rand = random.Random()
            rand.seed(config.model_te_data_shuffle_seed)
            rand.shuffle(texts)
        for text in texts:
            del text['text']
        logging.info(f'get final texts: {len(texts)}')
        random_check = random.randint(0, len(texts) - 11)
        logging.info(f'sample start from {random_check} of generated data: {texts[random_check: random_check + 10]}')
        return texts

    def find_answer_according_to_abcd_else_split(self, x):
        abcd = ["(A)", "(B)", "(C)", "(D)"]
        all_in = [0 if e in x else 1 for e in abcd]
        if sum(all_in) == 0:
            indexes = []
            for e in abcd:
                ind = x.find(e)
                assert ind != -1
                indexes.append(ind)
            max_lens = []
            for i, ind in enumerate(indexes[:-1]):
                max_lens.append(indexes[i + 1] - indexes[i])
            max_len = max(max_lens)
            ave_len = (indexes[-1] - indexes[0]) // 3
            indexes.append(indexes[-1] + max_len + (ave_len // 2))
            assert len(indexes) == 5
            answers = []
            for i, ind in enumerate(indexes[:-1]):
                answers.append(x[indexes[i]:indexes[i + 1]])
            assert len(answers) == 4
        else:
            ave_len = len(x) // 4
            answers = []
            for i in range(4):
                answers.append(x[ave_len * i: ave_len * (i + 1)])
        return answers

    def get_loader(self, config: TaskEmbConfig, model_config: TrainConfig):
        tokenizer_model = self.get_auto_tokenizer(config.to_learn_model_lm_name)
        tokenizer_te_model = self.get_auto_tokenizer(config.task_emb_model_lm_name)
        if config.to_learn_model_lm_name.startswith("gpt"):
            tokenizer_model.add_special_tokens({'pad_token': "*"})
            logging.warning(f'get pad token {tokenizer_model.pad_token} and pad token id {tokenizer_model.pad_token_id}')
        if config.task_emb_model_lm_name.startswith("gpt"):
            tokenizer_te_model.add_special_tokens({'pad_token': "*"})
            logging.warning(f'get pad token {tokenizer_te_model.pad_token} and pad token id {tokenizer_te_model.pad_token_id}')
        new_data_lines = self.get_new_data_lines(config)
        logging.info(f'get data size: {len(new_data_lines)}')

        def my_collate_fn(batch):
            texts = batch
            inp_length = len(texts[0])
            para = dict(
                text=texts,
                padding='longest',
                truncation='longest_first',
                max_length=config.max_len,
                return_tensors='pt',
                return_token_type_ids=True,
                return_attention_mask=True,
            )
            if inp_length == 1:
                para.update(dict(
                    text=[x[0] for x in texts],
                ))
                model_tokens = tokenizer_model(**para)
                te_model_token = tokenizer_te_model(**para)
            elif inp_length == 2:
                first_texts = [x[0] for x in texts]
                second_texts = [x[1] for x in texts]
                para.update(dict(
                    text=first_texts,
                    text_pair=second_texts,
                ))
                model_tokens = tokenizer_model(**para)
                te_model_token = tokenizer_te_model(**para)
            else:
                raise Exception(f'inp_length incorrect {inp_length}')
            return model_tokens.input_ids, model_tokens.token_type_ids, model_tokens.attention_mask, te_model_token.input_ids, te_model_token.token_type_ids, te_model_token.attention_mask

        dataset = CrossfitDataSetV1(config=config, model_config=model_config, data=new_data_lines)
        data_loader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=my_collate_fn, shuffle=False)
        return dataset, data_loader


class CrossfitDataSetV1(Dataset):
    def __init__(self,
                 config: TaskEmbConfig,
                 model_config: TrainConfig,
                 data,
                 ):
        self.config = config
        self.model_config = model_config
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        if self.config.task_type == 'cr':
            inp = TextClassificationSchema.DATASET_SCHEMA[self.model_config.to_train_dataset]['in']
            if len(inp) == 1:
                texts = [data[-1]]
            elif len(inp) == 2 and len(data) == 1:
                texts = [data[0], '']
            elif len(inp) == 2 and len(data) == 2:
                texts = data
            else:
                raise Exception(f'Invalid inp len: {len(inp)} and data len: {len(data)}')
        elif self.config.task_type == 'qa':
            if len(data) == 1:
                texts = [data[0], '']
            elif len(data) == 2:
                texts = data
        else:
            raise Exception(f'invalid task type: {self.config.task_type}')
        return texts