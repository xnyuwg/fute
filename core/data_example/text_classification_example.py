class TextClassificationSchema:
    DATASET_NAME_GLUE_ALL = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']
    DATASET_NAME_GLUE_CLS = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'wnli']
    DATASET_NAME_GLUE_REG = ['stsb']
    DATASET_NAME_NON_GLUE_CLS = ['snli', 'scitail']
    DATASET_SCHEMA = {
        'cola': {
            'read': ['glue', 'cola'],
            'in': ['sentence'],
            'label': ['unacceptable', 'acceptable'],
            'metric': 'matthews_correlation'
        },
        'mnli': {
            'read': ['glue', 'mnli'],
            'in': ['premise', 'hypothesis'],
            'label': ['entailment', 'neutral', 'contradiction'],
            'dev_name': 'validation_matched',
            'test_name': 'test_matched',
            'metric': 'accuracy'
        },
        'mrpc': {
            'read': ['glue', 'mrpc'],
            'in': ['sentence1', 'sentence2'],
            'label': ['not_equivalent', 'equivalent'],
            'metric': 'f1'
        },
        'qnli': {
            'read': ['glue', 'qnli'],
            'in': ['question', 'sentence'],
            'label': ['entailment', 'not_entailment'],
            'metric': 'accuracy'
        },
        'qqp': {
            'read': ['glue', 'qqp'],
            'in': ['question1', 'question2'],
            'label': ['not_duplicate', 'duplicate'],
            'metric': 'f1'
        },
        'rte': {
            'read': ['glue', 'rte'],
            'in': ['sentence1', 'sentence1'],
            'label': ['entailment', 'not_entailment'],
            'metric': 'accuracy'
        },
        'sst2': {
            'read': ['glue', 'sst2'],
            'in': ['sentence'],
            'label': ['negative', 'positive'],
            'metric': 'accuracy'
        },
        'stsb': {
            'read': ['glue', 'stsb'],
            'in': ['sentence1', 'sentence1'],
            'label': [],
            'type': 'reg',
            'metric': 'spearmanr'
        },
        'wnli': {
            'read': ['glue', 'wnli'],
            'in': ['sentence1', 'sentence1'],
            'label': ['entailment', 'not_entailment'],
            'metric': 'accuracy'
        },
        'snli': {
            'read': ['snli'],
            'in': ['premise', 'hypothesis'],
            'label': ['entailment', 'neutral', 'contradiction'],
            'metric_read': ['glue', 'mnli'],
            'metric': 'accuracy'
        },
        'scitail': {
            'read': ['scitail', 'tsv_format'],
            'in': ['premise', 'hypothesis'],
            'label': ['neutral', 'entails'],
            'metric_read': ['glue', 'mnli'],
            'metric': 'accuracy'
        },
        'boolq': {
            'read': ['super_glue', 'boolq'],
            'in': ['question', 'passage'],
            'label': ['False', 'True'],
            'metric': 'accuracy'
        },
        'cb': {
            'read': ['super_glue', 'cb'],
            'in': ['premise', 'hypothesis'],
            'label': ['entailment', 'contradiction', 'neutral'],
            'metric': 'f1'
        },
        'anli': {
            'read': ['anli'],
            'in': ['premise', 'hypothesis'],
            'label': ['entailment', 'neutral', 'contradiction'],
            'train_name': 'train_r3',
            'dev_name': 'dev_r3',
            'test_name': 'test_r3',
            'metric_read': ['super_glue', 'cb'],
            'metric': 'f1'
        },
        'wanli': {
            'read': ["alisawuffles/WANLI"],
            'in': ['premise', 'hypothesis'],
            'label': ['entailment', 'neutral', 'contradiction'],
            'label_name': 'gold',
            'dev_name': 'test',
            'metric_read': ['super_glue', 'cb'],
            'metric': 'f1'
        },
        'imdb': {
            'read': ["imdb"],
            'in': ['text'],
            'label': ['negative', 'positive'],
            'dev_name': 'test',
            'metric_read': ['glue', 'sst2'],
            'metric': 'accuracy'
        },
        'rotten_tomatoes': {
            'read': ["rotten_tomatoes"],
            'in': ['text'],
            'label': ['negative', 'positive'],
            'metric_read': ['glue', 'sst2'],
            'metric': 'accuracy'
        },
        'financial_phrasebank': {
            'read': ["financial_phrasebank", 'sentences_allagree'],
            'in': ['sentence'],
            'label': ['neutral', 'positive', 'negative'],
            'dev_name': None,
            'test_name': None,
            'metric_read': ['glue', 'sst2'],
            'metric': 'accuracy'
        },
        'medqa': {
            'read': ['medalpaca/medical_meadow_medqa'],
            'in': ['input', 'instruction'],
            'label': ['A', 'B', 'C', 'D', 'E'],
            'label_name': 'output',
            'dev_name': None,
            'test_name': None,
            'metric_read': ['super_glue', 'boolq'],
            'metric': 'accuracy',
            'label_process_fn': lambda x: x[0]
        },
        'medical_questions_pairs': {
            'read': ['medical_questions_pairs'],
            'in': ['question_1', 'question_2'],
            'label': ['otherwise', 'similar'],
            'dev_name': None,
            'test_name': None,
            'metric_read': ['glue', 'mrpc'],
            'metric': 'f1'
        },
    }
    for dn, dd in DATASET_SCHEMA.items():
        if 'train_name' not in dd:
            dd['train_name'] = 'train'
        if 'dev_name' not in dd:
            dd['dev_name'] = 'validation'
        if 'test_name' not in dd:
            dd['test_name'] = 'test'
        if 'type' not in dd:
            dd['type'] = 'cls'
        if 'metric_read' not in dd:
            dd['metric_read'] = dd['read']
        if 'label_name' not in dd:
            dd['label_name'] = 'label'
