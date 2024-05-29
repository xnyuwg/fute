import logging
from core.data_processor.basic_processor import BasicProcessor
from datasets import load_dataset_builder, load_dataset
from core.data_example.text_classification_example import TextClassificationSchema
import copy


class TextClassificationProcessor(BasicProcessor):
    def __init__(self):
        super().__init__()

    def read_one_data(self, dn):
        """
        data = {
            'read':
            'in':
            'label':
            'type': cls / reg
            'train_name':
            'dev_name':
            'test_name':
            'data': [train, dev, test]
        }
        train/dev/test = {
            'input': {
                ...
            }
            'output': {
                'label_num':
                'label_str':
            }
        }
        """
        logging.info(f'reading {dn}')
        data_schema = copy.deepcopy(TextClassificationSchema.DATASET_SCHEMA[dn])
        label_name = data_schema['label_name']
        label_id2str = data_schema['label']
        label_str2id = {x: i for i, x in enumerate(label_id2str)}
        dataset = load_dataset(*data_schema['read'])
        split_sets = []
        for split_name in [data_schema['train_name'], data_schema['dev_name'], data_schema['test_name']]:
            if split_name is None:
                split_sets.append(None)
                continue
            split_data = []
            for ins in dataset[split_name]:
                example = {'input': {}, 'output': {}}
                for inp in data_schema['in']:
                    example['input'][inp] = ins[inp]
                if data_schema['type'] == 'cls':
                    if 'label_process_fn' in data_schema:
                        ins[label_name] = data_schema['label_process_fn'](ins[label_name])
                    if ins[label_name] in label_str2id:
                        example['output']['label_str'] = ins[label_name]
                        example['output']['label_num'] = label_str2id[ins[label_name]]
                    elif ins[label_name] == -1:
                        example['output']['label_num'] = None
                        example['output']['label_str'] = None
                    else:
                        assert ins[label_name] in list(range(len(label_id2str)))
                        example['output']['label_num'] = ins[label_name]
                        example['output']['label_str'] = label_id2str[ins[label_name]]
                elif data_schema['type'] == 'reg':
                    example['output']['label_num'] = ins[label_name]
                else:
                    raise Exception(f"data type {data_schema['type']} incorrect")
                split_data.append(example)
            split_sets.append(split_data)
        data_schema['data'] = split_sets
        return data_schema


if __name__ == '__main__':
    import importlib
    import logging
    importlib.reload(logging)
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')
    gp = TextClassificationProcessor()