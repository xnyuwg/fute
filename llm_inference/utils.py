import os
import json
import yaml
from datasets import load_dataset


CUR_DIR = os.path.dirname(os.path.abspath(__file__))


class FileIO():
    @staticmethod
    def save_json(file: dict, fpath: str) -> None:
        with open(fpath, 'w') as f:
            json.dump(file, f, indent=4)
        f.close()

    @staticmethod
    def load_txt(fpath: str) -> str:
        with open(fpath, 'r') as f:
            data = f.read()
        f.close()

        return data

    @staticmethod
    def load_json(fpath: str) -> dict:
        with open(fpath, 'r') as f:
            data = json.load(f)
        f.close()

        return data

    @staticmethod
    def load_jsonl(fpath: str) -> list:
        new_data = []

        with open(fpath, 'r') as f:
            raw_data = f.readlines()
            for line in raw_data:
                cur_line = json.loads(line)
                new_data.append(cur_line)

        return new_data
    
    @staticmethod
    def load_yaml(fpath: str) -> dict:
        with open(fpath, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

        return data


class DataUtils:
    
    file_io = FileIO()

    task_load_config = {
        'anli': ('anli', None, 'test_r3'),
        'cb': ('super_glue', 'cb', 'validation'),
        'wanli': ('alisawuffles/WANLI', None, 'test'),
        'sst2': ('glue', 'sst2', 'validation'),
        'imdb': ('imdb', None, 'test'),
        'rotten_tomatoes': ('rotten_tomatoes', None, 'test'),
    }

    @staticmethod
    def load_data(dataset_name: str):
        task_info = DataUtils.task_load_config[dataset_name]
        if task_info[1]:
            return load_dataset(task_info[0], task_info[1], split=task_info[2])
        else:
            return load_dataset(task_info[0], split=task_info[2])

    @staticmethod
    def load_model_te_data(path: str):
        data = DataUtils.file_io.load_jsonl(path)
        return data

    @staticmethod
    def preprocess_data(dataset_name: str, data: dict):
        if dataset_name == 'anli':
            out = {
                'premise': data['premise'],
                'hypothesis': data['hypothesis'],
                'answer': data['label']
            }
        elif dataset_name == 'cb':
            out = {
                'premise': data['premise'],
                'hypothesis': data['hypothesis'],
                'answer': data['label']
            }
        elif dataset_name == 'wanli':
            out = {
                'premise': data['premise'],
                'hypothesis': data['hypothesis'],
                'answer': data['gold']
            }
        elif dataset_name == 'sst2':
            out = {
                'text': data['sentence'],
                'answer': data['label']
            }
        elif dataset_name == 'imdb':
            out = {
                'text': data['text'],
                'answer': data['label']
            }
        elif dataset_name == 'rotten_tomatoes':
            out = {
                'text': data['text'],
                'answer': data['label']
            }
        return out

    @staticmethod
    def preprocess_model_te_data(task: str, data: dict):
        if task == 'nli':
            out = {
                'premise': data['premise'],
                'hypothesis': data['hypothesis'],
                'answer': 'none'
            }
        elif task == 'sa':
            out = {
                'text': data['content'],
                'answer': 'none'
            }
        return out

    @staticmethod
    def parse_number_string(s):
        parts = s.split(',')
        numbers = []
        for part in parts:
            if '-' in part:
                start, end = map(int, part.split('-'))
                numbers.extend(range(start, end + 1))
            else:
                numbers.append(int(part))
        return numbers


class PromptUtils:

    fileio = FileIO()
    data2task = {
        'anli': 'nli',
        'cb': 'nli',
        'wanli': 'nli',
        'race': 'mcqa',
        'cosmos_qa': 'mcqa',
        'sst2': 'sa',
        'imdb': 'sa',
        'rotten_tomatoes': 'sa',
    }

    def __init__(self):
        self.system_prompt = self.fileio.load_txt('./llm_inference/prompts/system_prompts/system.txt')
        self.instruction_postfix = self.fileio.load_txt('./llm_inference/prompts/system_prompts/instruction.txt')
        self.cot_postfix = self.fileio.load_txt('./llm_inference/prompts/system_prompts/cot.txt')

    def load_prompts(self, data_name: str, spec_task_name=None) -> tuple:
        data_name = data_name.lower().strip()
        if spec_task_name is None:
            task_name = self.data2task[data_name]
        else:
            task_name = spec_task_name

        templates = self.fileio.load_json(f'./llm_inference/prompts/{task_name}_use.json')

        return templates, task_name

    def load_model_te_prompts(self, task_name: str) -> dict:
        templates = self.fileio.load_json(f'./llm_inference/prompts/{task_name}_use.json')
        return templates

    def load_cot_prompts(self):
        new_instruction = self.fileio.load_txt('./llm_inference/prompts/system_prompts/cot_followup.txt')
        return new_instruction

    def apply_nli_template(self, data: dict, template: str, label_spaces: dict, cot: bool) -> list:
        if cot:
            template += '\n' + self.cot_postfix
        else:
            template += '\n' + self.instruction_postfix
        assert '{{premise}}' in template and '{{hypothesis}' in template
        filled_template = template.replace('{{premise}}', data['premise']) \
            .replace('{{hypothesis}}', data['hypothesis']) \
            .replace('{{label_space}}', ', '.join([ele.strip() for v in label_spaces.values() for ele in v]))
        cur_message = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': filled_template}
        ]
        return cur_message

    def apply_sa_template(self, data: dict, template: str, label_spaces: dict, cot: bool) -> list:
        if cot:
            template += '\n' + self.cot_postfix
        else:
            template += '\n' + self.instruction_postfix
        assert '{{text}}' in template
        filled_template = template.replace('{{text}}', data['text']) \
            .replace('{{label_space}}', ', '.join([ele.strip() for v in label_spaces.values() for ele in v]))
        cur_message = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': filled_template}
        ]
        return cur_message

    def apply_qa_template(self, data: dict, template: str, verbalizers: list, cot: bool) -> list:
        if cot:
            verbalizers = ', '.join(verbalizers[:-1]) + ', or ' + verbalizers[-1]
            self.cot_postfix = self.cot_postfix.replace('{{verbalizers}}', verbalizers)
            cur_question = data['question'] + '\n' + self.cot_postfix
        else:
            cur_question = data['question'] + '\n' + self.instruction_postfix
        filled_template = template.replace('{{ context }}', data['context']) \
            .replace('{{ question }}', cur_question.strip()) \
            .replace('{{ answer0 }}', data['options'][0]) \
            .replace('{{ answer1 }}', data['options'][1]) \
            .replace('{{ answer2 }}', data['options'][2]) \
            .replace('{{ answer3 }}', data['options'][3]) \
            .replace('{{label_space}}', 'A, B, C, or D') \
            .split('|||')[0] \
            .replace('{{', '') \
            .replace('}}', '').strip()

        cur_message = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': filled_template}
        ]

        return cur_message
