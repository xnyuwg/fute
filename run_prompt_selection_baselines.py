import argparse
from abc import ABC, abstractmethod
import numpy
from core.data_example.text_classification_example import TextClassificationSchema
import scipy
import evaluate
import sklearn
import sklearn.cluster
from core.utils.util_data import UtilData
from core.utils.util_math import UtilMath
from tqdm import tqdm


class AbstractPromptSelection(ABC):
    @abstractmethod
    def get_rank_score(self, prompts_data):
        """
        prompts_data: [prompt1_js, prompt2_js, ..., promptN_js]
        """
        pass

    def process_one_data(self, js):
        data = []
        for one in js['results']:
            ds = {
                'task': js['task'],
                'dataset': js['dataset'],
                'llm_model': js['model'],
                'prob': one['normalized_prob'],
                'pred': one['normalized_pred'],
                'answer': one['normalized_answer'],
                'ppl': one['prompt_ppl']
            }
            data.append(ds)
        return data

    def process_data(self, prompts_data):
        return [self.process_one_data(x) for x in prompts_data]


class MiPromptSelection(AbstractPromptSelection):
    def get_rank_score(self, prompts_data):
        prompts_data = self.process_data(prompts_data)
        results = []
        for prompt_data in prompts_data:
            probs = [x['prob'] for x in prompt_data]
            probs = numpy.array(probs)
            assert len(probs.shape) == 2
            # I = H(Y) - H(Y|X)
            # H(Y)
            hy_probs = probs.mean(axis=0)
            assert len(hy_probs.shape) == 1
            hy = scipy.stats.entropy(hy_probs)
            assert len(hy.shape) == 0 or (len(hy.shape) == 1 and len(hy) == 1)
            hy = hy.item()
            # H(Y|X)
            hyx = scipy.stats.entropy(probs, axis=1)
            assert len(hyx.shape) == 1
            hyx = hyx.mean()
            assert len(hyx.shape) == 0 or (len(hyx.shape) == 1 and len(hyx) == 1)
            hyx = hyx.item()
            ixy = hy - hyx
            results.append(ixy)
        return results


class SpellPromptSelection(AbstractPromptSelection):
    def get_rank_score(self, prompts_data):
        prompts_data = self.process_data(prompts_data)
        results = []
        for prompt_data in prompts_data:
            ppls = [x['ppl'] for x in prompt_data]
            ppl = sum(ppls) / len(ppls)
            results.append(ppl)
        results = [-x for x in results]
        return results


class GlobalEntropyPromptSelection(AbstractPromptSelection):
    def get_rank_score(self, prompts_data):
        prompts_data = self.process_data(prompts_data)
        results = []
        for prompt_data in prompts_data:
            if prompt_data[0]['task'] == 'nli':
                pred_count = [0, 0, 0]
            elif prompt_data[0]['task'] == 'sa':
                pred_count = [0, 0]
            for one in prompt_data:
                pred_count[one['pred']] += 1
            pred_count = numpy.array(pred_count)
            pred_count = pred_count / len(prompt_data)
            ge = scipy.stats.entropy(pred_count)
            results.append(ge)
        return results


class LocalEntropyPromptSelection(AbstractPromptSelection):
    def get_rank_score(self, prompts_data):
        prompts_data = self.process_data(prompts_data)
        results = []
        for prompt_data in prompts_data:
            probs = [x['prob'] for x in prompt_data]
            probs = numpy.array(probs)
            assert len(probs.shape) == 2
            le = scipy.stats.entropy(probs, axis=1)
            assert len(le.shape) == 1
            le = le.mean()
            assert len(le.shape) == 0 or (len(le.shape) == 1 and len(le) == 1)
            le = le.item()
            results.append(le)
        return results


class MdlEntropyPromptSelection(AbstractPromptSelection):
    def get_rank_score(self, prompts_data):
        prompts_data = self.process_data(prompts_data)
        results = []
        for prompt_data in prompts_data:
            probs = [x['prob'] for x in prompt_data]
            probs = numpy.array(probs)
            assert len(probs.shape) == 2
            e = -numpy.log(probs)
            e = e.mean(axis=1)
            e = e.mean()
            e = e.item()
            results.append(e)
        return results


class ZpsPromptSelection(AbstractPromptSelection):
    @abstractmethod
    def get_ensemble_prob_fn(self, probs):
        pass

    def get_ensemble_score(self, ensemble_prompts_data):
        data_len = len(ensemble_prompts_data[0])
        for prompt_data in ensemble_prompts_data:
            assert len(prompt_data) == data_len
        pseudo_answers = []
        for i in range(data_len):
            probs = [prompt_data[i]['prob'] for prompt_data in ensemble_prompts_data]
            probs = numpy.array(probs)
            assert len(probs.shape) == 2
            prob = self.get_ensemble_prob_fn(probs)
            assert len(prob.shape) == 1
            pseudo_answer = numpy.argmax(prob)
            pseudo_answer = pseudo_answer.item()
            pseudo_answers.append(pseudo_answer)
        return pseudo_answers

    def filter_prompts(self, prompts_data):
        prompts_conf = []
        for prompt_data in prompts_data:
            prompt_conf = []
            for one in prompt_data:
                prob = one['prob']
                prob_sort = sorted(prob, reverse=True)
                conf = prob_sort[0] - prob_sort[1]
                prompt_conf.append(conf)
            mean_conf = sum(prompt_conf) / len(prompt_conf)
            prompts_conf.append(mean_conf)
        prompts_conf = numpy.array(prompts_conf).reshape(-1, 1)
        kmeans = sklearn.cluster.KMeans(n_clusters=2, random_state=0, n_init=10).fit(prompts_conf)
        kmeans_index = kmeans.cluster_centers_.argmax()
        prompts_idxes = numpy.where(kmeans.labels_ == kmeans_index)[0]
        filtered_prompts_data = [prompts_data[x] for x in prompts_idxes]
        return filtered_prompts_data

    def get_rank_score(self, prompts_data):
        accuracy_evaluate = evaluate.combine(["accuracy"])
        prompts_data = self.process_data(prompts_data)
        ensemble_prompts_data = self.filter_prompts(prompts_data)
        pseudo_answer = self.get_ensemble_score(ensemble_prompts_data)
        results = []
        for prompt_data in prompts_data:
            preds = [x['pred'] for x in prompt_data]
            acc = accuracy_evaluate.compute(predictions=preds, references=pseudo_answer)['accuracy']
            results.append(acc)
        return results


class ZpsLogProbabilityMeanPromptSelection(ZpsPromptSelection):
    def get_ensemble_prob_fn(self, probs):
        probs = numpy.log(probs)
        prob = numpy.mean(probs, axis=0)
        return prob


class ZpsProbabilityMeanPromptSelection(ZpsPromptSelection):
    def get_ensemble_prob_fn(self, probs):
        prob = numpy.mean(probs, axis=0)
        return prob


class ZpsMajorityVotePromptSelection(ZpsPromptSelection):
    def get_ensemble_prob_fn(self, probs):
        prob = numpy.argmax(probs, axis=1)
        assert len(prob.shape) == 1
        res = numpy.zeros(probs.shape[1])
        for x in prob:
            res[x] += 1
        return res


class SelfSelectPromptSelection(AbstractPromptSelection):
    def __init__(self):
        self.rank_dict = {
            'llama-13b': {
                'nli': [10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],
                'sa': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            },
            'llama-70b': {
                'nli': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                'sa': [11, 4, 3, 9, 0, 5, 1, 10, 6, 7, 12, 2, 8],
            },
            'mixtral': {
                'nli': [6, 9, 2, 4, 3, 8, 7, 10, 1, 5, 0, 11, 12],
                'sa': [9, 5, 2, 4, 1, 3, 6, 10, 7, 8, 12, 0, 11],
            },
        }

    def get_rank_score(self, prompts_data):
        prompts_data = self.process_data(prompts_data)
        llm_model = prompts_data[0][0]['llm_model']
        task = prompts_data[0][0]['task']
        results = self.rank_dict[llm_model][task][:len(prompts_data)]
        return results


class PromptSelectionMethodManager:
    def __init__(self):
        self.methods = {
            'mi': MiPromptSelection(),
            'spell': SpellPromptSelection(),
            'ge': GlobalEntropyPromptSelection(),
            'le': LocalEntropyPromptSelection(),
            'mdl': MdlEntropyPromptSelection(),
            'zlpm': ZpsLogProbabilityMeanPromptSelection(),
            'zpm': ZpsProbabilityMeanPromptSelection(),
            'zmv': ZpsMajorityVotePromptSelection(),
            'selfsel': SelfSelectPromptSelection()
        }
        self.files = {}
        self.nli_datasets = ['cb', 'anli', 'wanli']
        self.sa_datasets = ['sst2', 'imdb', 'rotten_tomatoes']
        self.nli_id2label = ['entailment', 'neutral', 'contradiction']
        self.nli_label2id = {x: i for i, x in enumerate(self.nli_id2label)}
        self.sa_id2label = ['negative', 'positive']
        self.sa_label2id = {x: i for i, x in enumerate(self.sa_id2label)}
        self.datasets_to_task_type = {x: 'nli' if x in self.nli_datasets else 'sa' for x in self.nli_datasets + self.sa_datasets}

    def read_file(self, path):
        path_str = str(path)
        if path_str in self.files:
            return self.files[path_str]
        else:
            if path_str.endswith('.json'):
                file_content = UtilData.read_raw_json_file(path_str, verbose=False)
            elif path_str.endswith('.jsonl'):
                file_content = UtilData.read_raw_jsonl_file(path_str, verbose=False)
            self.files[path_str] = file_content
            return file_content

    def get_label_to_id_id_to_label(self, dataset):
        if self.datasets_to_task_type[dataset] == 'nli':
            return self.nli_label2id, self.nli_id2label
        elif self.datasets_to_task_type[dataset] == 'sa':
            return self.sa_label2id, self.sa_id2label

    def normalize_data(self, js):
        dataset = js['dataset']
        label2id, id2label = self.get_label_to_id_id_to_label(dataset)
        answer_choices = js['prompt_info']['answer_choices']
        verb_to_label = {v: k for k, vs in answer_choices.items() for v in vs}
        data = js['results']
        for one in data:
            token_info = one['token_info']
            word_scores = token_info['word_scores']
            verbalizers = token_info['verbalizers']
            final_score_dict = {k: 0 for k in answer_choices}
            for ws, vb in zip(word_scores, verbalizers):
                pred_name = verb_to_label[vb]
                pred_score = ws
                if pred_score > final_score_dict[pred_name]:
                    final_score_dict[pred_name] = ws
            scores = [final_score_dict[ln] for ln in answer_choices.keys()]
            prob = UtilMath.numpy_softmax(numpy.array(scores)).tolist()
            assert 0.99 < sum(prob) < 1.01
            max_index = numpy.argmax(one['token_info']['word_scores'])
            pred_label = one['token_info']['verbalizers'][max_index]
            pred_label = verb_to_label[pred_label]
            pred = label2id[pred_label]
            answer = one['data_entry']['answer']
            if answer in list(range(3)):
                answer = TextClassificationSchema.DATASET_SCHEMA[dataset]['label'][answer]
            answer = label2id[answer]
            one['normalized_answer'] = answer
            one['normalized_pred'] = pred
            one['normalized_prob'] = prob
        return js

    def get_all_rank_scores(self, paths):
        """
        :param paths: the path of results of prompt 1-13
        """
        final_results = {}
        for method_name, method_object in tqdm(self.methods.items()):
            prompts_data = []
            for path in paths:
                js = self.read_file(path)
                js = self.normalize_data(js)
                prompts_data.append(js)
            pred_rank_scores = method_object.get_rank_score(prompts_data)
            final_results[method_name] = pred_rank_scores
        return final_results


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--paths", type=str, required=False, default='False')
    arg_parser.add_argument("--save_path", type=str, required=False, default='False')
    args = arg_parser.parse_args()
    args.paths = args.paths.split(',')
    if len(args.paths) != 13:
        print(f'Warning! the number of paths != 13')
    print(f'a total of {len(args.paths)} path')
    if not args.save_path.endswith('.json'):
        args.save_path += '.json'

    psmm = PromptSelectionMethodManager()
    result = psmm.get_all_rank_scores(args.paths)
    UtilData.write_json_file(args.save_path, result)
