import random
import argparse
import importlib
import logging
import torch
import time
from task_emb_2023.conf.train_conf import TrainConfig
from task_emb_2023.conf.task_emb_conf import TaskEmbConfig
from task_emb_2023.model.task_emb_prefix_model import TaskEmbModelPrefixV1
from task_emb_2023.model.task_emb_fim_model import TaskEmbModelFimV1
from task_emb_2023.model.cr_model import CrModelV1
from task_emb_2023.model.qa_model import QaModelV1
from core.data_example.text_classification_example import TextClassificationSchema
from core.optimizer.basic_optimizer import BasicOptimizer
from task_emb_2023.trainer.task_emb_trainer import TaskEmbTrainer
from core.conf.global_config_manager import GlobalConfigManager
from core.utils.util_string import UtilString
from task_emb_2023.data_process.cr_preparer import CrPreparer
from task_emb_2023.data_process.qa_preparer import QaPreparer
from task_emb_2023.data_process.crossfit_preparer import CrossfitPreparer
from core.data_processor.crossfit_processor import CrossfitProcessor
from core.utils.util_data import UtilData
from task_emb_2023.data_process.mte_read_preparer import MteReadPreparer
import multiprocessing
importlib.reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')


def get_debug_read_list():
    x = [
        'glue_cola',
        'glue_mnli',
        'glue_mrpc',
        'glue_qnli',
        'glue_qqp',
        'glue_rte',
        'glue_sst2',
        'glue_wnli',
        'scitail',
        'squad',
        'hotpot_qa',
        'duorc',
        'boolq',
    ]
    x = [
        'trec_finegrained',
        'wiki_auto',
        'eli5',
    ]
    x = [
        'glue_cola',
        'glue_mrpc',
        'glue_wnli',
    ]
    x = [
    ]
    return x


def get_exclude_read_list_transfer():
    x = [
        'glue_cola',
        'glue_mnli',
        'glue_mrpc',
        'glue_qnli',
        'glue_qqp',
        'glue_rte',
        'glue_sst2',
        'glue_wnli',
        'scitail',
        'squad',
        'hotpot_qa',
        'duorc',
        'boolq',
    ]
    return x


def get_exclude_read_list_llm(config):
    if config.task_type == 'nli':
        x = [
            'superglue_cb',
            'anli',
        ]
    elif config.task_type == 'mcqa':
        x = [
            'sciq',
            'cosmos_qa',
            'race',
        ]
    elif config.task_type == 'sa':
        x = [
            'glue_sst2',
            'imdb',
            'rotten_tomatoes'
        ]
    else:
        raise NotImplementedError(f'unknown task_emb_type: {config.task_type}')
    return x


def get_read_list():
    x = [
        'acronym_identification',
        'ade_classification',
        'ade_dosage',
        'ade_effect',
        'adversarial_qa',
        'aeslc',
        'agnews',
        'ai2_arc',
        'amazon_polarity',
        'anli',
        'app_reviews',
        'aqua_rat',
        'art',
        'aslg_pc12',
        'biomrc',
        'blimp',
        'boolq',
        'break',
        'circa',
        'climate_fever',
        'codah',
        'commongen',
        'commonsense_qa',
        'cos_e',
        'cosmos_qa',
        'crawl_domain',
        'crows_pairs',
        'dbpedia_14',
        'definite_pronoun_resolution',
        'discovery',
        'dream',
        'duorc',
        'e2e_nlg_cleaned',
        'emo',
        'emotion',
        'empathetic_dialogues',
        'ethos',
        'financial_phrasebank',
        'freebase_qa',
        'gigaword',
        'glue_cola',
        'glue_mnli',
        'glue_mrpc',
        'glue_qnli',
        'glue_qqp',
        'glue_rte',
        'glue_sst2',
        'glue_wnli',
        'google_wellformed_query',
        'hate_speech18',
        'hate_speech_offensive',
        'hatexplain',
        'health_fact',
        'hellaswag',
        'hotpot_qa',
        'imdb',
        'jeopardy',
        'kilt_ay2',
        'kilt_fever',
        'kilt_hotpotqa',
        'kilt_nq',
        'kilt_trex',
        'kilt_wow',
        'kilt_zsre',
        'lama',
        'liar',
        'limit',
        'math_qa',
        'mc_taco',
        'medical_questions_pairs',
        'mocha',
        'multi_news',
        'numer_sense',
        'onestop_english',
        'openbookqa',
        'paws',
        'piqa',
        'poem_sentiment',
        'proto_qa',
        'qa_srl',
        'qasc',
        'quail',
        'quarel',
        'quartz',
        'quoref',
        'race',
        'reddit_tifu',
        'ropes',
        'rotten_tomatoes',
        'samsum',
        'scicite',
        'sciq',
        'scitail',
        'search_qa',
        'sick',
        'sms_spam',
        'social_i_qa',
        'spider',
        'squad',
        'superglue_cb',
        'superglue_copa',
        'superglue_multirc',
        'superglue_record',
        'superglue_rte',
        'superglue_wic',
        'superglue_wsc',
        'swag',
        'tab_fact',
        'trec',
        'tweet_eval',
        'tweet_qa',
        'web_questions',
        'wiki_bio',
        'wiki_qa',
        'wiki_split',
        'wikisql',
        'winogrande',
        'wiqa',
        'xsum',
        'yahoo_answers_topics',
        'yelp_polarity',
        'yelp_review_full'
    ]
    return x


def get_config(args) -> TaskEmbConfig:
    config = TaskEmbConfig()

    config.task_type = args.task_type
    config.input_type = args.input_type
    config.task_emb_type = args.task_emb_type
    config.to_learn_model_lm_name = args.lm_model_name
    config.task_emb_model_lm_name = args.task_emb_model_lm_name
    config.is_baseline = args.is_baseline
    config.llm_mode = args.llm_mode
    config.to_learn_model_dataset = args.to_train_dataset
    config.save_path = args.save_path
    config.load_path = args.load_path
    if not config.save_path.endswith('.npy'):
        config.save_path += '.npy'

    config.prefix_len = 16

    if config.input_type == 'model':
        config.max_data_for_task_emb_train = 1000 if args.max_data_for_task_emb_train is None else args.max_data_for_task_emb_train
        config.max_epochs = 1 if args.max_epochs is None else args.max_epochs
        config.early_stop = False
        config.early_stop_patient = 0
        if config.task_type == 'cr':
            config.max_len = 128
        elif config.task_type == 'qa':
            config.max_len = 384
        real_batch_size = 128
        if config.task_emb_type.startswith('prefix'):
            config.gradient_accumulation_steps = 16 if args.gradient_accumulation_steps is None else args.gradient_accumulation_steps
            learning_rate = 1e-3 if args.learning_rate is None else args.learning_rate
        elif config.task_emb_type.startswith('fim'):
            config.gradient_accumulation_steps = 32 if args.gradient_accumulation_steps is None else args.gradient_accumulation_steps
            learning_rate = 2e-5 if args.learning_rate is None else args.learning_rate

        if config.llm_mode:
            config.max_len = 512
            config.max_len -= config.prefix_len
    elif config.input_type == 'data' and config.task_type == 'cr':
        config.max_len = 128
        config.gradient_accumulation_steps = 1 if args.gradient_accumulation_steps is None else args.gradient_accumulation_steps
        real_batch_size = 32
        if config.is_baseline:
            config.early_stop = False
            config.early_stop_patient = 0
            if config.task_emb_type.startswith('prefix'):
                learning_rate = 1e-2 if args.learning_rate is None else args.learning_rate
                config.max_epochs = 60 if args.max_epochs is None else args.max_epochs
            elif config.task_emb_type.startswith('fim'):
                learning_rate = 2e-5 if args.learning_rate is None else args.learning_rate
                config.max_epochs = 3 if args.max_epochs is None else args.max_epochs
        else:
            config.max_epochs = 120 if args.max_epochs is None else args.max_epochs
            config.early_stop = True
            config.early_stop_patient = 4
            if config.task_emb_type.startswith('prefix'):
                learning_rate = 1e-2 if args.learning_rate is None else args.learning_rate
            elif config.task_emb_type.startswith('fim'):
                learning_rate = 2e-5 if args.learning_rate is None else args.learning_rate
            config.model_save_path = config.cache_path + config.save_path.split('.npy')[0]
            GlobalConfigManager.if_not_exist_then_creat(config.model_save_path)
    elif config.input_type == 'data' and config.task_type == 'qa':
        config.max_len = 384
        config.gradient_accumulation_steps = 1 if args.gradient_accumulation_steps is None else args.gradient_accumulation_steps
        real_batch_size = 32
        if config.is_baseline:
            config.early_stop = False
            config.early_stop_patient = 0
            if config.task_emb_type.startswith('prefix'):
                learning_rate = 5e-3 if args.learning_rate is None else args.learning_rate
                config.max_epochs = 30 if args.max_epochs is None else args.max_epochs
            elif config.task_emb_type.startswith('fim'):
                learning_rate = 3e-5 if args.learning_rate is None else args.learning_rate
                config.max_epochs = 3 if args.max_epochs is None else args.max_epochs
        else:
            config.max_epochs = 60 if args.max_epochs is None else args.max_epochs
            config.early_stop = True
            config.early_stop_patient = 4
            if config.task_emb_type.startswith('prefix'):
                learning_rate = 5e-3 if args.learning_rate is None else args.learning_rate
            elif config.task_emb_type.startswith('fim'):
                learning_rate = 3e-5 if args.learning_rate is None else args.learning_rate
            config.model_save_path = config.cache_path + config.save_path.split('.npy')[0]
            GlobalConfigManager.if_not_exist_then_creat(config.model_save_path)
    config.device = torch.device('cuda')
    config.data_loader_shuffle = False if args.data_loader_shuffle is None else args.data_loader_shuffle
    config.batch_size = real_batch_size // config.gradient_accumulation_steps
    config.early_stop_loss_window = 10
    config.learning_rate_slow = learning_rate
    config.learning_rate_fast = learning_rate
    config.scheduler_type = 'linear'
    config.model_te_data_shuffle_seed = 42
    return config


def config_set_model(args, config: TaskEmbConfig, model_config):
    config.save_path = args.save_path
    config.num_labels = model_config.num_labels


def get_cr_model_config(args, config: TaskEmbConfig) -> TrainConfig:
    model_config = TrainConfig()

    model_config.task_type = 'cr'
    model_config.lm_model_name = config.to_learn_model_lm_name
    model_config.pretrain_dataset = None
    model_config.to_train_dataset = config.to_learn_model_dataset

    model_config.model_save_path = args.load_path

    model_config.cr_to_train_schema = TextClassificationSchema.DATASET_SCHEMA[model_config.to_train_dataset]
    if model_config.cr_to_train_schema['type'] == 'reg':
        model_config.num_labels = 1
    elif model_config.cr_to_train_schema['type'] == 'cls':
        model_config.num_labels = len(model_config.cr_to_train_schema['label'])

    model_config.device = torch.device('cuda')
    model_config.data_loader_shuffle = False
    model_config.max_epochs = 3
    model_config.max_len = 128
    model_config.gradient_accumulation_steps = 1 if args.gradient_accumulation_steps is None else args.gradient_accumulation_steps
    model_config.batch_size = 32 // model_config.gradient_accumulation_steps
    learning_rate = 2e-5
    model_config.learning_rate_slow = learning_rate
    model_config.learning_rate_fast = learning_rate
    model_config.scheduler_type = 'linear'
    return model_config


def get_qa_model_config(args, config: TaskEmbConfig) -> TrainConfig:
    model_config = TrainConfig()

    model_config.task_type = 'qa'
    model_config.lm_model_name = config.to_learn_model_lm_name
    model_config.pretrain_dataset = None
    model_config.to_train_dataset = config.to_learn_model_dataset

    model_config.model_save_path = args.load_path

    model_config.device = torch.device('cuda')
    model_config.data_loader_shuffle = False
    model_config.max_epochs = 3
    model_config.max_len = 384
    model_config.gradient_accumulation_steps = 1 if args.gradient_accumulation_steps is None else args.gradient_accumulation_steps
    model_config.batch_size = 32 // model_config.gradient_accumulation_steps
    learning_rate = 3e-5
    model_config.learning_rate_slow = learning_rate
    model_config.learning_rate_fast = learning_rate
    model_config.scheduler_type = 'linear'
    return model_config


def get_all_config(args):
    config = get_config(args)
    model_config = None
    if config.input_type == 'model' and config.llm_mode:
        if config.task_type == 'nli':
            config.num_labels = 3
        elif config.task_type == 'sa':
            config.num_labels = 2
        config.save_path = args.save_path
    elif config.input_type == 'model' and config.task_type == 'cr':
        model_config = get_cr_model_config(args, config)
        config_set_model(args, config, model_config)
        assert not config.is_baseline
    elif config.input_type == 'model' and config.task_type == 'qa':
        model_config = get_qa_model_config(args, config)
        config_set_model(args, config, model_config)
        assert not config.is_baseline
    elif config.input_type == 'data' and config.task_type == 'cr':
        model_config = None
        config.to_train_dataset = config.to_learn_model_dataset
        config.lm_model_name = config.task_emb_model_lm_name
        config.cr_to_train_schema = TextClassificationSchema.DATASET_SCHEMA[config.to_learn_model_dataset]
        if config.cr_to_train_schema['type'] == 'reg':
            config.num_labels = 1
        elif config.cr_to_train_schema['type'] == 'cls':
            config.num_labels = len(config.cr_to_train_schema['label'])
        config.save_path = args.save_path
    elif config.input_type == 'data' and config.task_type == 'qa':
        model_config = None
        config.to_train_dataset = config.to_learn_model_dataset
        config.lm_model_name = config.task_emb_model_lm_name
        config.save_path = args.save_path
    return config, model_config


def run_one(args, crossfit_pre=None):
    config, model_config = get_all_config(args)
    logging.info(f' start new run_one save path={config.save_path}, tmb_type={config.task_emb_type}')

    task_emb_model = None
    if config.task_emb_type.startswith('prefix'):
        task_emb_model = TaskEmbModelPrefixV1(config)
    elif config.task_emb_type.startswith('fim'):
        task_emb_model = TaskEmbModelFimV1(config)

    model = None
    if config.input_type == 'model' and config.task_type == 'cr':
        model = CrModelV1(model_config)
        model.load_model(model_config.model_save_path + '/torch_checkpoint.pt')
    elif config.input_type == 'model' and config.task_type == 'qa':
        model = QaModelV1(model_config)
        model.load_model(model_config.model_save_path + '/torch_checkpoint.pt')

    datasets, dataloaders = None, None
    if config.input_type == 'model':
        dataset, dataloader = crossfit_pre.get_loader(config=config, model_config=model_config)
        datasets = [dataset, None]
        dataloaders = [dataloader, None]
    elif config.input_type == 'data' and config.task_type == 'cr':
        pre = CrPreparer(config)
        datasets, dataloaders = pre.get_loader()
    elif config.input_type == 'data' and config.task_type == 'qa':
        pre = QaPreparer(config)
        datasets, dataloaders = pre.get_loader(config.to_learn_model_dataset)

    if config.input_type == 'model' and (not config.llm_mode):
        model.to(config.device)
    task_emb_model.to(config.device)

    optimizer = BasicOptimizer(config=config,
                               model=task_emb_model,
                               )

    trainer = TaskEmbTrainer(config=config,
                             model_config=model_config,
                             task_emb_model=task_emb_model,
                             model=model,
                             optimizer=optimizer,
                             train_loader=dataloaders[0],
                             dev_loader=dataloaders[1],
                             dev_dataset=datasets[1],
                             )

    trainer.train()


def run_mana(args):
    config = get_config(args)
    if config.input_type == 'model':
        start_time = time.time()
        if config.llm_mode:
            crossfit_pre = MteReadPreparer(lm_model_name=config.task_emb_model_lm_name)
        else:
            if args.model_te_read_num_processes < 2:
                logging.info(f'model_te_read_num_processes {args.model_te_read_num_processes}, default using debug mode')
                pro = CrossfitProcessor(max_data_num=config.max_data_for_task_emb_train, choose_seed=config.model_te_data_shuffle_seed)
                data_lines, _ = pro.read_data(get_debug_read_list(), input_num=(1, 2))
                crossfit_pre = CrossfitPreparer(data_lines=data_lines, lm_model_name=config.task_emb_model_lm_name)
            else:
                num_processes = args.model_te_read_num_processes
                read_list = get_read_list()[:]
                if args.generate_model_data_only:
                    exclude_list = set(get_exclude_read_list_llm(config)[:])
                else:
                    exclude_list = set(get_exclude_read_list_transfer()[:])
                read_list = [x for x in read_list if x not in exclude_list]
                logging.info(f'total read_list: {len(read_list)}')
                random.shuffle(read_list)
                chunk_size = len(read_list) // num_processes
                chunks = [read_list[i:i + chunk_size] for i in range(0, len(read_list), chunk_size)]
                with multiprocessing.Pool(num_processes) as pool:
                    results = pool.starmap(crossfit_worker_function, [(chunk, (config.max_data_for_task_emb_train, config.model_te_data_shuffle_seed)) for chunk in chunks])
                data_lines = [line for sublist in results for line in sublist]
                crossfit_pre = CrossfitPreparer(data_lines=data_lines, lm_model_name=config.task_emb_model_lm_name)
            used_time = (time.time() - start_time) / 60
            logging.info('read all multiprocessing done used_time={:.2f}'.format(used_time))
    else:
        crossfit_pre = None
    if args.generate_model_data_only:
        assert config.input_type.startswith('model')
        logging.info(f'generate_model_data_only...')
        data_to_save = crossfit_pre.generate_model_data_entry(config=config, max_data_num=10000)
        save_path = args.save_path
        if not save_path.endswith('.jsonl'):
            save_path += '.jsonl'
        UtilData.write_jsonl_file(save_path, data_to_save)
    else:
        run_one(args, crossfit_pre)


def crossfit_worker_function(chunk, para):
    pro = CrossfitProcessor(max_data_num=para[0], choose_seed=para[1])
    data_lines, _ = pro.read_data(chunk, input_num=(1, 2))
    return data_lines


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--save_path", type=str, required=True)
    arg_parser.add_argument("--load_path", type=str, required=False, default=None)
    arg_parser.add_argument("--task_type", type=str, required=True, default=None)
    arg_parser.add_argument("--input_type", type=str, required=True, default=None)
    arg_parser.add_argument("--task_emb_type", type=str, required=True, default=None)
    arg_parser.add_argument("--to_train_dataset", type=str, required=False, default=None)
    arg_parser.add_argument("--lm_model_name", type=str, required=True)
    arg_parser.add_argument("--task_emb_model_lm_name", type=str, required=False, default='t5-base')
    arg_parser.add_argument("--learning_rate", type=float, required=False, default=None)
    arg_parser.add_argument("--max_epochs", type=int, required=False, default=None)
    arg_parser.add_argument("--gradient_accumulation_steps", type=int, required=False, default=None)
    arg_parser.add_argument("--max_data_for_task_emb_train", type=int, required=False, default=None)
    arg_parser.add_argument("--is_baseline", type=str, required=False, default="False")
    arg_parser.add_argument("--model_te_read_num_processes", type=int, required=False, default=10)
    arg_parser.add_argument("--generate_model_data_only", type=str, required=False, default="False")
    arg_parser.add_argument("--llm_mode", type=str, required=False, default="False")
    arg_parser.add_argument("--data_loader_shuffle", type=str, required=False, default=None)
    args = arg_parser.parse_args()
    args.is_baseline = UtilString.str_to_bool(args.is_baseline)
    args.generate_model_data_only = UtilString.str_to_bool(args.generate_model_data_only)
    args.llm_mode = UtilString.str_to_bool(args.llm_mode)
    args.data_loader_shuffle = UtilString.str_to_bool(args.data_loader_shuffle) if args.data_loader_shuffle is not None else None
    return args


if __name__ == '__main__':
    args = parse_args()
    run_mana(args)
