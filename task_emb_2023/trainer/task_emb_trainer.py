from typing import List, Callable
import logging
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from core.model.basic_model import BasicModel
import torch
from core.optimizer.basic_optimizer import BasicOptimizer
from task_emb_2023.conf.train_conf import TrainConfig
from task_emb_2023.conf.task_emb_conf import TaskEmbConfig
from task_emb_2023.trainer.train_trainer import TrainTrainer
from datasets import Dataset


class TaskEmbTrainer:
    def __init__(self,
                 config: TaskEmbConfig,
                 model_config: TrainConfig,
                 task_emb_model: BasicModel,
                 optimizer: BasicOptimizer,
                 train_loader: DataLoader,
                 dev_loader: DataLoader = None,
                 model: BasicModel = None,
                 dev_dataset: Dataset = None,
                 ):
        self.config = config
        self.model_config = model_config
        self.model = model
        self.task_emb_model: BasicModel = task_emb_model
        self.optimizer: BasicOptimizer = optimizer
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.dev_dataset = dev_dataset
        self.data_trainer = None
        self.model_task_emb_run_time = 0

        num_training_steps = (len(train_loader) * config.max_epochs) // config.gradient_accumulation_steps + 1
        if self.config.input_type == 'model' and (self.config.task_type == 'sa' or self.config.task_type == 'nli') and self.config.llm_mode:
            assert self.model is None
            self.model_run_fn = self.llm_model_run
            self.optimizer.prepare_for_train(num_training_steps=num_training_steps, gradient_accumulation_steps=config.gradient_accumulation_steps)
        elif self.config.input_type == 'model' and self.config.task_type == 'cr':
            assert self.model is not None
            self.model.eval()
            self.model_run_fn = self.cr_model_run
            self.optimizer.prepare_for_train(num_training_steps=num_training_steps, gradient_accumulation_steps=config.gradient_accumulation_steps)
        elif self.config.input_type == 'model' and self.config.task_type == 'qa':
            assert self.model is not None
            self.model.eval()
            self.model_run_fn = self.qa_model_run
            self.optimizer.prepare_for_train(num_training_steps=num_training_steps, gradient_accumulation_steps=config.gradient_accumulation_steps)
        elif self.config.input_type == 'data' and self.config.task_type == 'cr':
            assert self.model is None
            self.data_trainer = TrainTrainer(config=config,
                                             model=self.task_emb_model,
                                             optimizer=optimizer,
                                             train_loader=train_loader,
                                             dev_loader=dev_loader,
                                             )
        elif self.config.input_type == 'data' and self.config.task_type == 'qa':
            assert self.model is None
            self.data_trainer = TrainTrainer(config=config,
                                             model=self.task_emb_model,
                                             optimizer=optimizer,
                                             train_loader=train_loader,
                                             dev_loader=dev_loader,
                                             dev_dataset=dev_dataset
                                             )

    def cr_model_run(self, batch):
        model_input_ids, model_token_type_ids, model_attention_mask, te_model_input_ids, te_model_token_type_ids, te_model_attention_mask = (b.to(self.config.device) for b in batch)
        with torch.no_grad():
            _, pred = self.model(input_ids=model_input_ids,
                                 token_type_ids=model_token_type_ids,
                                 attention_mask=model_attention_mask,
                                 )

        if self.model_config.cr_to_train_schema['type'] == 'cls':
            assert len(pred.shape) == 2
            pred = torch.softmax(pred, dim=1)
        pred = pred.detach()
        start_time = time.time()
        loss, _ = self.task_emb_model(input_ids=te_model_input_ids,
                                      token_type_ids=te_model_token_type_ids,
                                      attention_mask=te_model_attention_mask,
                                      labels=pred,
                                      )
        used_time = (time.time() - start_time) / 60
        self.model_task_emb_run_time += used_time
        return loss

    def qa_model_run(self, batch):
        model_input_ids, model_token_type_ids, model_attention_mask, te_model_input_ids, te_model_token_type_ids, te_model_attention_mask = (b.to(self.config.device) for b in batch)
        with torch.no_grad():
            _, start_pred, end_pred = self.model(input_ids=model_input_ids,
                                                 token_type_ids=model_token_type_ids,
                                                 attention_mask=model_attention_mask,
                                                 )

        assert len(start_pred.shape) == 2
        assert len(end_pred.shape) == 2
        start_pred = start_pred.detach()
        end_pred = end_pred.detach()
        start_pred = torch.argmax(start_pred, dim=1)
        end_pred = torch.argmax(end_pred, dim=1)
        start_time = time.time()
        loss, _, _ = self.task_emb_model(input_ids=te_model_input_ids,
                                         token_type_ids=te_model_token_type_ids,
                                         attention_mask=te_model_attention_mask,
                                         start_positions=start_pred,
                                         end_positions=end_pred,
                                         )
        used_time = (time.time() - start_time) / 60
        self.model_task_emb_run_time += used_time
        return loss

    def llm_model_run(self, batch):
        te_model_input_ids, te_model_token_type_ids, te_model_attention_mask, te_model_label = (b.to(self.config.device) for b in batch)
        start_time = time.time()
        loss, _ = self.task_emb_model(input_ids=te_model_input_ids,
                                      token_type_ids=te_model_token_type_ids,
                                      attention_mask=te_model_attention_mask,
                                      labels=te_model_label,
                                      )
        used_time = (time.time() - start_time) / 60
        self.model_task_emb_run_time += used_time
        return loss

    def train_with_model(self, do_fim=False):
        self.model_task_emb_run_time = 0
        self.task_emb_model.train()
        if do_fim:
            self.task_emb_model.eval()
        backward_time = 0
        loss_records = []
        patience = self.config.early_stop_patient
        epoch_loss = None
        max_epochs = self.config.max_epochs
        if do_fim:
            max_epochs = 1
        for epoch in range(max_epochs):
            start_time = time.time()
            with tqdm(self.train_loader, unit="b", position=0, leave=True) as tqdm_epoch:
                for batch in tqdm(self.train_loader):
                    loss = self.model_run_fn(batch)
                    backward_start_time = time.time()
                    if not do_fim:
                        self.optimizer.gradient_update(loss)
                    else:
                        loss.backward()
                    backward_time += (time.time() - backward_start_time) / 60
                    step_loss = loss.item()
                    epoch_loss = step_loss if epoch_loss is None else 0.98 * epoch_loss + 0.02 * step_loss
                    tqdm_epoch.set_postfix(ls=epoch_loss, pa=patience)
                    loss_records.append(step_loss)
                    this_window = loss_records[- self.config.early_stop_loss_window:]
                    prev_window = loss_records[- self.config.early_stop_loss_window * 2: - self.config.early_stop_loss_window]
                    this_mean_loss = sum(this_window) / len(this_window) if len(this_window) > 0 else -1
                    prev_mean_loss = sum(prev_window) / len(prev_window) if len(prev_window) > 0 else -1
                    if prev_mean_loss < this_mean_loss and len(loss_records) > self.config.early_stop_loss_window * 2:
                        patience -= 1
                    if self.config.early_stop and patience <= 0:
                        break
            used_time = (time.time() - start_time) / 60
            logging.info('Total Train Time = {:.4f} min'.format(used_time))
        logging.info("Train End")

    def train_with_dataset(self):
        self.data_trainer.train(False, False)

    def fim_with_dataset(self):
        self.data_trainer.fim_run()

    def train(self):
        if self.config.input_type == 'model':
            self.train_with_model()
            if self.config.task_emb_type.startswith('fim'):
                logging.info(f"getting fim embeddings")
                self.optimizer.optimizer.zero_grad()
                self.task_emb_model.init_task_embedding_run()
                self.train_with_model(do_fim=True)
        elif self.config.input_type == 'data':
            self.train_with_dataset()
            if self.config.task_emb_type.startswith('fim'):
                logging.info(f"getting fim embeddings")
                self.optimizer.optimizer.zero_grad()
                self.task_emb_model.init_task_embedding_run()
                self.fim_with_dataset()
        else:
            raise Exception(f'Unsupported input type: {self.config.input_type}')
        task_emb = self.task_emb_model.get_task_embedding()
        return task_emb
