import logging
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from core.model.basic_model import BasicModel
import evaluate
from core.optimizer.basic_optimizer import BasicOptimizer
from core.utils.util_data import UtilData
import numpy
from task_emb_2023.conf.train_conf import TrainConfig
from datasets import Dataset
import collections


class TrainTrainer:
    def __init__(self,
                 config: TrainConfig,
                 model: BasicModel,
                 optimizer: BasicOptimizer,
                 train_loader: DataLoader,
                 dev_loader: DataLoader,
                 dev_dataset: Dataset = None,
                 ):
        self.config = config
        self.model: BasicModel = model
        self.optimizer: BasicOptimizer = optimizer
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.dev_dataset = dev_dataset
        num_training_steps = (len(train_loader) * config.max_epochs) // config.gradient_accumulation_steps + 1
        self.optimizer.prepare_for_train(num_training_steps=num_training_steps,
                                         gradient_accumulation_steps=config.gradient_accumulation_steps)
        if self.config.task_type == 'cr':
            self.metric = evaluate.load(*self.config.cr_to_train_schema['metric_read'])
            self.model_run_fn = self.cr_model_run
            self.compute_metric = self.cr_compute_metric
        elif self.config.task_type == 'qa':
            self.metric = evaluate.load('squad_v2')
            self.model_run_fn = self.qa_model_run
            self.compute_metric = self.qa_compute_metric
        else:
            raise ValueError(f'unknown task type {self.config.task_type}')

    def eval(self,
             dataloader: DataLoader,
             dataset: Dataset = None,
             run_eval=True,
             epoch=-1,
             ):
        self.model.eval()
        start_time = time.time()
        raw_results = []
        for batch in tqdm(dataloader, unit="b", position=0, leave=True):
            loss, res = self.model_run_fn(batch, run_eval=run_eval)
            raw_results.append(res)
        score_result = self.compute_metric(raw_results, dataset)
        used_time = (time.time() - start_time) / 60
        logging.info('Eval Epoch = {}, Time = {:.2f} min, Score = {}'.format(epoch, used_time, score_result))
        return score_result, raw_results

    def fim_run(self,
                ):
        self.model.eval()
        start_time = time.time()
        for batch in tqdm(self.train_loader, unit="b", position=0, leave=True):
            loss, res = self.model_run_fn(batch, run_eval=False)
            loss.backward()
        used_time = (time.time() - start_time) / 60
        logging.info('FIM Time = {:.2f}'.format(used_time))

    def train(self,
              save_score=True,
              save_model=True):
        if save_score:
            score_results_file_name = self.config.score_save_path
        best_score = 0
        best_score_results = None
        patient = self.config.early_stop_patient
        for epoch in range(self.config.max_epochs):
            self.model.train()
            start_time = time.time()
            batch_step = 0
            epoch_loss = None
            with tqdm(self.train_loader, unit="b", position=0, leave=True) as tqdm_epoch:
                for batch in tqdm(self.train_loader):
                    batch_step += 1
                    loss, res = self.model_run_fn(batch, run_eval=False)
                    self.optimizer.gradient_update(loss)
                    epoch_loss = loss.item() if epoch_loss is None else 0.98 * epoch_loss + 0.02 * loss.item()
                    tqdm_epoch.set_postfix(ls=epoch_loss)
            used_time = (time.time() - start_time) / 60
            logging.info('Train Epoch = {}, Time = {:.2f} min, Epoch Mean Loss = {:.4f}'.format(epoch, used_time, epoch_loss))
            dev_score_results, _ = self.eval(dataloader=self.dev_loader, dataset=self.dev_dataset, epoch=epoch)
            if self.config.early_stop:
                current_score = dev_score_results['best']
                if current_score >= best_score:
                    self.model.save_model(self.config.model_save_path + '/torch_model_best.pt')
                    best_score = current_score
                    best_score_results = dev_score_results
                    patient = self.config.early_stop_patient
                else:
                    patient -= 1
                if self.config.early_stop and patient <= 0:
                    break

        if self.config.early_stop:
            self.model.load_model(self.config.model_save_path + '/torch_model_best.pt')
        if save_score:
            if self.config.early_stop:
                final_dev_score_results = best_score_results
            else:
                final_dev_score_results, _ = self.eval(dataloader=self.dev_loader, dataset=self.dev_dataset, epoch=self.config.max_epochs)
            to_save_final_dev_score_results = {key: float(value) if isinstance(value, numpy.float64) else value for key, value in final_dev_score_results.items()}
            UtilData.write_json_file(score_results_file_name, to_save_final_dev_score_results)
        if self.config.model_save_path is not None and save_model:
            self.model.save_lm_model()
            self.model.save_model(self.config.model_save_path + '/torch_checkpoint.pt')
        logging.info("Train End")

    def cr_model_run(self, batch, run_eval):
        input_ids, token_type_ids, attention_mask, labels = (b.to(self.config.device) for b in batch)
        model_res = self.model(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               labels=labels,
                               )
        loss, result = model_res
        out_record = {
            'loss': loss.item(),
            'pred': result.detach().cpu().numpy(),
            'label': batch[-1].detach().cpu().numpy(),
        }
        return loss, out_record

    def cr_compute_metric(self, raw_results, dataset):
        losses = [x['loss'] for x in raw_results]
        mean_loss = sum(losses) / len(losses) if len(losses) > 0 else -1

        preds = [x['pred'] for x in raw_results]
        preds = numpy.concatenate(preds, axis=0)
        labels = [x['label'] for x in raw_results]
        labels = numpy.concatenate(labels, axis=0)
        if self.config.cr_to_train_schema['type'] == 'cls':
            preds = numpy.argmax(preds, axis=1)

        scores = self.metric.compute(predictions=preds, references=labels)
        scores['all'] = numpy.mean(list(scores.values()))
        scores['loss'] = mean_loss
        scores['best'] = scores[self.config.cr_to_train_schema['metric']]
        return scores

    def qa_model_run(self, batch, run_eval):
        index = batch[0]
        input_ids, token_type_ids, attention_mask, start_positions, end_positions = (b.to(self.config.device) for b in batch[1:])
        model_res = self.model(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               start_positions=start_positions,
                               end_positions=end_positions,
                               )
        loss, pred_start, pred_end = model_res
        out_record = {
            'loss': loss.item(),
            'index': index.detach().cpu().numpy(),
            'pred_start': pred_start.detach().cpu().numpy(),
            'pred_end': pred_end.detach().cpu().numpy(),
            'ans_start': start_positions.detach().cpu().numpy(),
            'ans_end': end_positions.detach().cpu().numpy(),
        }
        return loss, out_record

    def qa_compute_metric(self, raw_results, datasets):
        dataset, inputs = datasets

        losses = [x['loss'] for x in raw_results]
        mean_loss = sum(losses) / len(losses) if len(losses) > 0 else -1

        index = [x['index'] for x in raw_results]
        index = numpy.concatenate(index, axis=0)
        pred_start = [x['pred_start'] for x in raw_results]
        pred_start = numpy.concatenate(pred_start, axis=0)
        pred_end = [x['pred_end'] for x in raw_results]
        pred_end = numpy.concatenate(pred_end, axis=0)
        ans_start = [x['ans_start'] for x in raw_results]
        ans_start = numpy.concatenate(ans_start, axis=0)
        ans_end = [x['ans_end'] for x in raw_results]
        ans_end = numpy.concatenate(ans_end, axis=0)

        scores = self.qa_compute_metrics(start_logits=pred_start, end_logits=pred_end, features=inputs, examples=dataset)

        for key in ['exact', 'f1', 'HasAns_exact', 'HasAns_f1', 'NoAns_exact', 'NoAns_f1', 'best_exact', 'best_f1']:
            if key in scores:
                scores[key] = scores[key] / 100

        scores['best'] = scores['f1']
        scores['loss'] = mean_loss
        return scores

    def qa_compute_metrics(self, start_logits, end_logits, features, examples):
        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features["example_id"]):
            example_to_features[feature].append(idx)

        predicted_answers = []
        for example in examples:
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features["post_offset_mapping"][feature_index]
                start_indexes = numpy.argsort(start_logit)[-1 : (0 - self.config.qa_n_best_size - 1) : -1].tolist()
                end_indexes = numpy.argsort(end_logit)[-1 : (0 - self.config.qa_n_best_size - 1) : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > self.config.qa_max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0] : offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"], "no_answer_probability": 0.0}
                )
            else:
                predicted_answers.append({"id": example_id, "prediction_text": "", "no_answer_probability": 1.0})

        theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return self.metric.compute(predictions=predicted_answers, references=theoretical_answers)
