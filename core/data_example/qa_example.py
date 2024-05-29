class QaSchema:
    DATASET_NAME_ALL = ['squad', 'squad_v2']
    DATASET_SCHEMA = {
        'squad': {
            'read': ['squad'],
            'in': ['id', 'question', 'context'],
            'answer': ['text', 'answer_start'],
        },
        'squad_v2': {
            'read': ['squad_v2'],
            'in': ['id', 'question', 'context'],
            'answer': ['text', 'answer_start'],
            'has_no_answer': True,
        },
    }
    for dn, dd in DATASET_SCHEMA.items():
        if 'train_name' not in dd:
            dd['train_name'] = 'train'
        if 'dev_name' not in dd:
            dd['dev_name'] = 'validation'
        if 'test_name' not in dd:
            dd['test_name'] = 'test'
        if 'has_no_answer' not in dd:
            dd['has_no_answer'] = False

