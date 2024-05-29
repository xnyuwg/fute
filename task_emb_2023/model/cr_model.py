from core.model.basic_model import BasicModel
from transformers import AutoModelForSequenceClassification
from task_emb_2023.conf.train_conf import TrainConfig
import logging


class CrModelV1(BasicModel):
    def __init__(self,
                 config: TrainConfig,
                 ):
        super().__init__()
        self.config = config
        self.model_save_path = self.config.model_save_path + '/hf_checkpoint' if self.config.model_save_path is not None else None
        if self.config.load_path is None:
            self.model_load_path = None
            self.language_model = AutoModelForSequenceClassification.from_pretrained(self.config.lm_model_name, num_labels=self.config.num_labels)
        else:
            self.model_load_path = self.config.load_path + '/hf_checkpoint'
            self.language_model = AutoModelForSequenceClassification.from_pretrained(self.model_load_path,
                                                                                     num_labels=self.config.num_labels,
                                                                                     ignore_mismatched_sizes=True,
                                                                                     problem_type=None)
        if self.config.lm_model_name.startswith('gpt'):
            self.language_model.config.pad_token_id = 9

    def save_lm_model(self):
        if self.model_save_path is None:
            raise ValueError('saving when model_save_path is none')
        self.language_model.save_pretrained(self.model_save_path)
        logging.info("Saved language model at {}".format(self.model_save_path))

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                labels=None):

        if self.config.lm_model_name.startswith('t5'):
            lm_output = self.language_model(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            labels=labels)
        elif self.config.lm_model_name.startswith('bert') or self.config.lm_model_name.startswith('gpt'):
            lm_output = self.language_model(input_ids=input_ids,
                                            token_type_ids=token_type_ids,
                                            attention_mask=attention_mask,
                                            labels=labels)
        else:
            raise NotImplementedError(f'get lm name {self.config.lm_model_name}')

        if self.config.gradient_accumulation_steps is not None and self.config.gradient_accumulation_steps > 1 and lm_output.loss is not None:
            lm_output.loss = lm_output.loss / self.config.gradient_accumulation_steps

        return lm_output.loss, lm_output.logits

