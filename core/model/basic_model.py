import logging
from torch import nn
from transformers import BertModel, BartModel, BartForConditionalGeneration, RobertaForTokenClassification, BertForTokenClassification, AutoModel
import torch.nn.utils
from core.conf.global_config_manager import GlobalConfigManager
from typing import List
import copy
from scipy.optimize import linear_sum_assignment
import math
import torch.nn.functional as F
from core.conf.basic_conf import BasicConfig
from transformers import AutoTokenizer, PreTrainedTokenizer


class BasicModel(nn.Module):
    o_token = 'O'
    b_suffix = '-B'
    i_suffix = '-I'

    @staticmethod
    def new_bert_model(model_name: str = 'bert-base-uncased') -> BertModel:
        bert = BertModel.from_pretrained(model_name, cache_dir=GlobalConfigManager.get_transformers_cache_path())
        return bert

    @staticmethod
    def new_bart_model(model_name: str) -> BartModel:
        bart = BartModel.from_pretrained(model_name, cache_dir=GlobalConfigManager.get_transformers_cache_path())
        return bart

    @staticmethod
    def new_bart_model_conditional_generation(model_name: str) -> BartForConditionalGeneration:
        bart = BartForConditionalGeneration.from_pretrained(model_name, cache_dir=GlobalConfigManager.get_transformers_cache_path())
        return bart

    @staticmethod
    def new_roberta_for_token_classification_model(model_name, num_labels) -> BertModel:
        bert = RobertaForTokenClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir=GlobalConfigManager.get_transformers_cache_path())
        return bert

    @staticmethod
    def new_bert_for_token_classification_model(model_name, num_labels) -> BertModel:
        bert = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels, cache_dir=GlobalConfigManager.get_transformers_cache_path())
        return bert

    @staticmethod
    def new_auto_model(model_name: str) -> AutoModel:
        model = AutoModel.from_pretrained(model_name, cache_dir=GlobalConfigManager.get_transformers_cache_path())
        return model

    @staticmethod
    def find_BIO_spans_positions(spans: list) -> List[list]:
        """ return [[start, end], [start, end]] """
        o_token = BasicModel.o_token
        b_suffix = BasicModel.b_suffix
        if len(spans) == 0:
            return []

        # split according to O, this section is slowest
        to_read_spans = copy.deepcopy(spans)
        o_split_spans: List[list] = []
        left_index = 0
        while left_index < len(to_read_spans):
            for i in range(left_index, len(to_read_spans)):
                if to_read_spans[i] == o_token:
                    if i == left_index:
                        o_split_spans.append(to_read_spans[i:i + 1])
                        left_index = i + 1
                    else:
                        o_split_spans.append(to_read_spans[left_index:i])
                        left_index = i
                    break
                if i == len(to_read_spans) - 1:
                    o_split_spans.append(to_read_spans[left_index:i + 1])
                    left_index = i + 1
                    break
        # print(to_read_spans)
        # print(o_split_spans)
        assert sum([len(x) for x in o_split_spans]) == len(spans)

        # split according to B, this section
        bo_split_spans: List[list] = []
        for oss in o_split_spans:
            if len(oss) == 1:
                bo_split_spans.append(oss)
            else:
                to_read_oss = oss
                while len(to_read_oss) > 0:
                    if len(to_read_oss) == 1:
                        bo_split_spans.append(to_read_oss[:])
                        to_read_oss = []
                    for i in range(len(to_read_oss)):
                        if to_read_oss[i].endswith(b_suffix):
                            if i != 0:
                                bo_split_spans.append(to_read_oss[:i])
                                to_read_oss = to_read_oss[i:]
                                break
                        elif i == len(to_read_oss) - 1:
                            bo_split_spans.append(to_read_oss[:])
                            to_read_oss = []
                            break
        # print(bo_split_spans)
        assert sum([len(x) for x in bo_split_spans]) == len(spans)

        # split different care
        boc_split_spans: List[list] = []
        for boss in bo_split_spans:
            if len(boss) == 1:
                boc_split_spans.append(boss)
            else:
                to_read_boss = boss
                while len(to_read_boss) > 0:
                    if len(to_read_boss) == 1:
                        boc_split_spans.append(to_read_boss[:])
                        to_read_boss = []
                    for i in range(1, len(to_read_boss)):
                        if to_read_boss[i-1][:-2] != to_read_boss[i][:-2]:
                            boc_split_spans.append(to_read_boss[:i])
                            to_read_boss = to_read_boss[i:]
                            break
                        elif i == len(to_read_boss) - 1:
                            boc_split_spans.append(to_read_boss[:])
                            to_read_boss = []
                            break
        # print(boc_split_spans)
        assert sum([len(x) for x in boc_split_spans]) == len(spans)

        # extract positions
        positions = []
        start_len = 0
        for boss in boc_split_spans:
            end_len = start_len + len(boss)
            if len(boss) == 1 and boss[0] == o_token:
                pass
            else:
                positions.append([start_len, end_len])
            start_len = end_len
        # print(positions)

        return positions

    @staticmethod
    def validify_BIO_span(spans: List[str], positions: List[list], mode='ignore'):
        """
        mode:
        ignore: ignore the span without B
        modify: modify the span's first I into B
        """
        o_token = BasicModel.o_token
        b_suffix = BasicModel.b_suffix
        spans = copy.deepcopy(spans)
        for pos in positions:
            start, end = pos[0], pos[1]
            if not spans[start].endswith(b_suffix):
                if mode == 'ignore':
                    for i in range(start, end):
                        spans[i] = o_token
                elif mode == 'modify':
                    spans[start] = spans[start][:-2] + b_suffix
        return spans

    @staticmethod
    def find_most_similar_power_num(x: int):
        y = 1
        while y < x:
            y *= 2
        return y

    @staticmethod
    def event_ordering(matrix):
        cost = matrix
        try:
            row_ind, col_ind = linear_sum_assignment(cost, maximize=False)
        except ValueError:
            logging.error('event_ordering error with matrix: {}'.format(matrix))
            row_ind = [0]
            col_ind = [0]
        assert len(row_ind) == len(col_ind)
        res = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
        min_cost = cost[row_ind, col_ind].sum()
        return res, min_cost

    @staticmethod
    def get_embedding_tensor(word_num: int, embedding_size: int, word_to_index: dict, word_embedding_path, offset: int = 0, random_init: bool = False):
        logging.info("load word embedding from {} ...".format(str(word_embedding_path)))
        word_embedding_path = GlobalConfigManager.get_word_embedding_cache_path(word_embedding_path)
        if random_init:
            emb_tensor = nn.init.xavier_uniform_(torch.FloatTensor(word_num, embedding_size))
        else:
            emb_tensor = torch.zeros((word_num, embedding_size))
        count = 0
        with open(word_embedding_path, 'r') as file:
            first_line = True
            for line in file:
                items = line.split()
                if first_line:
                    first_line = False
                    continue
                word = items[0]
                emb = items[1:]
                if word in word_to_index:
                    assert len(emb) <= embedding_size
                    word_index = word_to_index[word]
                    emb = [float(x) for x in emb]
                    new_emb = torch.FloatTensor(emb)
                    emb_tensor[word_index, offset:offset+len(emb)] = new_emb
                    count += 1
        logging.info("load word embedding from {}, loaded {} word out of should {} as {}".format(str(word_embedding_path), count, word_num, count / word_num))
        return emb_tensor

    def __init__(self,
                 config: BasicConfig = None,
                 gradient_accumulation_steps: int = None):
        super().__init__()
        self.config = config
        self.slow_para = None
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.tokenizer = {}

    def get_model_device(self):
        return self.dummy_param.device

    def get_tokenizer(self, model_name: str = None) -> PreTrainedTokenizer:
        if model_name is None:
            model_name = self.config.lm_model_name
        if model_name not in self.tokenizer:
            logging.info("init {} auto tokenizer".format(model_name))
            tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=GlobalConfigManager.get_transformers_cache_path())
            self.tokenizer[model_name] = tokenizer
        return self.tokenizer[model_name]

    def save_model(self, path):
        logging.info('model save to {} ...'.format(path))
        model_state = self.state_dict()
        torch.save(model_state, path)

    def load_model(self, path):
        logging.info('model load from {} ...'.format(path))
        state = torch.load(path)
        self.load_state_dict(state)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''
    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss) if self.reduction == 'mean' else torch.sum(loss) if self.reduction == 'sum' else loss