from core.model.basic_model import BasicModel
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, GPT2ForSequenceClassification, T5ForSequenceClassification
from transformers import AutoModelForQuestionAnswering, T5ForQuestionAnswering, BertForQuestionAnswering, GPT2ForQuestionAnswering
from task_emb_2023.conf.task_emb_conf import TaskEmbConfig
import logging
import torch
import numpy


class TaskEmbModelFimV1(BasicModel):
    def __init__(self,
                 config: TaskEmbConfig,
                 ):
        super().__init__()
        self.config = config
        if self.config.task_type in ['cr', 'nli', 'sa']:
            self.task_emb_model = AutoModelForSequenceClassification.from_pretrained(config.task_emb_model_lm_name, num_labels=self.config.num_labels)
        elif self.config.task_type == 'qa':
            self.task_emb_model = AutoModelForQuestionAnswering.from_pretrained(config.task_emb_model_lm_name)

        if config.task_emb_model_lm_name.startswith('gpt'):
            self.task_emb_model.config.pad_token_id = 9

        self.computing_fim = False
        self.instance_count = None
        self.numerical_stable_value = 1e4

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                labels=None,
                start_positions=None,
                end_positions=None,
                ):
        batch_size = input_ids.shape[0]

        if self.config.task_type in ['cr', 'nli', 'sa']:
            if self.config.task_emb_model_lm_name.startswith('t5'):
                lm_output = self.task_emb_model(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                labels=labels,
                                                )
            elif self.config.task_emb_model_lm_name.startswith('bert') or self.config.task_emb_model_lm_name.startswith('gpt'):
                lm_output = self.task_emb_model(input_ids=input_ids,
                                                token_type_ids=token_type_ids,
                                                attention_mask=attention_mask,
                                                labels=labels,
                                                )
        elif self.config.task_type == 'qa':
            if self.config.task_emb_model_lm_name.startswith('t5'):
                lm_output = self.task_emb_model(input_ids=input_ids,
                                                attention_mask=attention_mask,
                                                start_positions=start_positions,
                                                end_positions=end_positions,
                                                )
            elif self.config.task_emb_model_lm_name.startswith('bert') or self.config.task_emb_model_lm_name.startswith('gpt'):
                lm_output = self.task_emb_model(input_ids=input_ids,
                                                token_type_ids=token_type_ids,
                                                attention_mask=attention_mask,
                                                start_positions=start_positions,
                                                end_positions=end_positions,
                                                )
        if not self.computing_fim:
            if self.config.gradient_accumulation_steps is not None and self.config.gradient_accumulation_steps > 1 and lm_output.loss is not None:
                lm_output.loss = lm_output.loss / self.config.gradient_accumulation_steps
        else:
            lm_output.loss = lm_output.loss * batch_size / self.numerical_stable_value
            self.instance_count += batch_size

        if self.config.task_type in ['cr', 'nli', 'sa']:
            return lm_output.loss, lm_output.logits
        elif self.config.task_type == 'qa':
            return lm_output.loss, lm_output.start_logits, lm_output.end_logits
        else:
            raise Exception(f'task type {self.config.task_type} incorrect')

    def init_task_embedding_run(self):
        self.computing_fim = True
        self.instance_count = 0

    def get_all_model_parameters(self, model):
        gradients = [param.grad.view(-1).detach().cpu() for param in model.parameters() if param.requires_grad]
        all_gradients = torch.cat(gradients)
        return all_gradients

    def get_task_embedding(self, save=True):
        if self.config.task_emb_model_lm_name.startswith('t5'):
            if self.config.task_type in ['cr', 'nli', 'sa']:
                the_model = self.task_emb_model.transformer
            elif self.config.task_type == 'qa':
                the_model = self.task_emb_model
            grads = torch.cat([
                self.get_all_model_parameters(the_model.shared),
                self.get_all_model_parameters(the_model.encoder),
                self.get_all_model_parameters(the_model.decoder)
            ], dim=0)
        elif self.config.task_emb_model_lm_name.startswith('bert'):
            grads = torch.cat([
                self.get_all_model_parameters(self.task_emb_model.bert.embeddings),
                self.get_all_model_parameters(self.task_emb_model.bert.encoder),
            ], dim=0)
            # grads = self.get_all_model_parameters(self.task_emb_model.bert)
        elif self.config.task_emb_model_lm_name.startswith('gpt'):
            grads = self.get_all_model_parameters(self.task_emb_model.transformer)
        grads = grads / self.instance_count * self.numerical_stable_value
        ave = torch.mean(grads)
        grads = torch.pow(grads, self.config.fim_pow)
        grads = grads.detach().cpu().numpy()
        path = self.config.save_path
        if save:
            numpy.save(path, grads)
        logging.info(f"get task embedding shape {grads.shape} with mean={ave.item()} (if) saving at {path}")
        return grads
