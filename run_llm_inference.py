import argparse
from tqdm import tqdm
from llm_inference.utils import FileIO, DataUtils, PromptUtils
from llm_inference.inference.llama import LlamaInference
from llm_inference.inference.mixtral import MixtralInference
import random


CONFIG_PATH_TABLE = {
    'vanilla': {
        'llama-13b': './llm_inference/configs/llama_vanilla_config.yaml',
        'llama-70b': './llm_inference/configs/llama_vanilla_config.yaml',
        'mixtral': './llm_inference/configs/mixtral_vanilla_config.yaml',
    },
    'cot': {
        'llama-13b': './llm_inference/configs/llama_cot_config.yaml',
        'llama-70b': './llm_inference/configs/llama_cot_config.yaml',
        'mixtral': './llm_inference/configs/mixtral_cot_config.yaml',
    }
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--prompt_number', type=int, required=True, help='prompt numbers >=10 using cot prompt')
    parser.add_argument('--model', type=str, required=True, help='Choose from [llama-13b, llama-70b, mixtral]')
    parser.add_argument('--dataset_name', type=str, required=True, help='Choose from [anli, cb, wanli, imdb, rotten_tomatoes, sst2]')
    parser.add_argument('--seed', type=int, default=42, required=False, help='Random seed')
    parser.add_argument('--bit', type=str, default='8', required=False, help='16, 8, 4')
    args = parser.parse_args()
    args.batch_size = 1
    return args


def main():
    args = get_args()
    data_utils = DataUtils()
    prompt_utils = PromptUtils()
    file_io = FileIO()

    if not args.save_path.endswith('.json'):
        args.save_path += '.json'
    args.system_prompt_path = './llm_inference/prompts/system.txt'

    print(f'starting... get model={args.model} with {args.bit}bit dataset_name={args.dataset_name} prompt_numbers={args.prompt_number}')

    assert args.bit in {'32', '16', '8', '4'}

    if 'llama' in args.model:
        model = LlamaInference()
        model.set_model(args.model)
        model.init_model(args.bit)
    elif 'mixtral' in args.model:
        model = MixtralInference()
        model.init_model(args.bit)

    dataset_name = args.dataset_name
    data = data_utils.load_data(dataset_name)
    prompt_templates, task_genre = prompt_utils.load_prompts(dataset_name)
    prompt_templates = prompt_templates[:] + prompt_templates[:]
    assert len(prompt_templates) == 20

    data_indexes = list(range(len(data)))
    if len(data) > 1000:
        rand = random.Random()
        rand.seed(args.seed)
        rand.shuffle(data_indexes)
        data_indexes = data_indexes[:args.used_data_size]
        data_indexes = sorted(data_indexes)
        data = data.select(data_indexes)
    assert len(data_indexes) == len(data)

    prompt_index = args.prompt_number
    assert prompt_index < 13
    use_cot = prompt_index >= 10
    if use_cot:
        print(f"processing... {prompt_index} is a CoT prompt")

    vanilla_config_path = CONFIG_PATH_TABLE['vanilla'][args.model]
    vanilla_config_dict = file_io.load_yaml(vanilla_config_path)

    if use_cot:
        new_instruction = prompt_utils.load_cot_prompts()
        cot_config_path = CONFIG_PATH_TABLE['cot'][args.model]
        cot_config_dict = file_io.load_yaml(cot_config_path)
        first_config, second_config = cot_config_dict, vanilla_config_dict
    else:
        new_instruction = None
        first_config, second_config = vanilla_config_dict, None

    prompt_dict = prompt_templates[prompt_index]
    cache_result = {
        'prompt_index': prompt_index,
        'prompt_info': prompt_dict,
        'task': task_genre,
        'dataset': dataset_name,
        'model': args.model,
        'bit': args.bit,
        'seed': args.seed,
        'variant': None,
        'cot': use_cot,
        'results': [],
    }

    input_batch = []
    entry_batch = []
    data_steps = []
    data_step = -1
    for entry in tqdm(data, position=1, leave=True):
        data_step += 1
        entry = data_utils.preprocess_data(dataset_name, entry)

        cur_verbalizers = [ele.strip() for v in prompt_dict['answer_choices'].values() for ele in v]
        if task_genre == 'nli':
            cur_message = prompt_utils.apply_nli_template(entry, prompt_dict['jinja'], prompt_dict['answer_choices'], use_cot)
        elif task_genre == 'sa':
            cur_message = prompt_utils.apply_sa_template(entry, prompt_dict['jinja'], prompt_dict['answer_choices'], use_cot)

        input_prompt = model.format_prompt(cur_message)

        input_batch.append(input_prompt)
        entry_batch.append(entry)
        data_steps.append(data_step)

        if len(input_batch) >= args.batch_size or data_step == len(data) - 1:
            assert len(input_batch) == len(entry_batch) == len(data_steps)

            result, token_probs, prompt_ppl, prompt_list, chat_history = model.inference(
                input_batch,
                cur_verbalizers,
                first_config,
                use_cot,
                stop_tokens=['\n', '\n\n'],
                new_instruction=new_instruction,
                second_round_config=second_config,
            )
            out_generated = [ele.split('[/INST]')[-1] for ele in result]
            for i in range(len(input_batch)):
                # NOTE: additionally save chat history for CoT
                if use_cot:
                    cache_result['results'].append({
                        'data_id': data_indexes[data_steps[i]],
                        'data_entry': entry_batch[i],
                        'input_promptt': prompt_list[i],
                        'chat_historyy': chat_history[i],
                        'token_info': token_probs[i],
                        'prompt_ppl': prompt_ppl[i],
                    })
                else:
                    cache_result['results'].append({
                        'data_id': data_indexes[data_steps[i]],
                        'data_entry': entry_batch[i],
                        'input_promptt': prompt_list[i],
                        'token_info': token_probs[i],
                        'prompt_ppl': prompt_ppl[i],
                    })
            input_batch = []
            entry_batch = []
            data_steps = []
    assert len(cache_result['results']) == len(data)

    file_io.save_json(cache_result, args.save_path)
    print(f'Done: {args.save_path}')


if __name__ == '__main__':
    main()

