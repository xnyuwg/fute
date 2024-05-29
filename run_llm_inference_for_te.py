import argparse
from tqdm import tqdm
from llm_inference.utils import FileIO, DataUtils, PromptUtils
from llm_inference.inference.llama import LlamaInference
from llm_inference.inference.mixtral import MixtralInference
import json
import os


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
    parser.add_argument('--data_load_path', type=str, required=True)
    parser.add_argument('--prompt_number', type=int, required=True, help='prompt numbers >=10 using cot prompt')
    parser.add_argument('--model', type=str, required=True, help='Choose from [llama-13b, llama-70b, mixtral]')
    parser.add_argument('--task_genre', type=str, required=True, help='nli or sa')
    parser.add_argument('--seed', type=int, default=42, required=False, help='Random seed')
    parser.add_argument('--bit', type=str, default=8, required=False, help='16, 8, 4')
    args = parser.parse_args()
    assert args.task_genre in ['nli', 'sa']
    args.batch_size = 1
    return args


def main():
    args = get_args()
    data_utils = DataUtils()
    prompt_utils = PromptUtils()
    file_io = FileIO()

    args.system_prompt_path = './llm_inference/prompts/system.txt'
    args.model_te_data_path = args.data_load_path
    if not args.save_path.endswith('.jsonl'):
        args.save_path += '.jsonl'

    print(f'starting... get model={args.model} with {args.bit}bit model_te_data_path={args.model_te_data_path} prompt_number={args.prompt_number}')

    assert int(args.bit) in {32, 16, 8, 4}

    if 'llama' in args.model:
        model = LlamaInference()
        model.set_model(args.model)
        model.init_model(args.bit)
    elif 'mixtral' in args.model:
        model = MixtralInference()
        model.init_model(args.bit)

    data = data_utils.load_model_te_data(args.model_te_data_path)
    prompt_templates = prompt_utils.load_model_te_prompts(args.task_genre)
    prompt_templates = prompt_templates[:] + prompt_templates[:]
    assert len(prompt_templates) == 20

    prompt_index = args.prompt_number
    use_cot = prompt_index >= 10
    path = args.save_path

    if os.path.exists(path):
        read_js = file_io.load_jsonl(path)
        processed_data_index = []
        for rj in read_js:
            processed_data_index.append(rj['data_index'])
        max_data_index = max(processed_data_index) if len(processed_data_index) > 0 else 0
        processed_data_index = set(processed_data_index)
        print(f'path existed! processed_data={len(processed_data_index)}, max_data_index={max_data_index}, reading from={path}')
    else:
        processed_data_index = None
        print(f'path does not exist!')

    file = open(path, "a")
    prompt_dict = prompt_templates[prompt_index]

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

    input_batch = []
    entry_batch = []
    data_steps = []
    data_step = -1
    skip_print = False
    for entry in tqdm(data, position=1, leave=True):
        data_step += 1
        if processed_data_index is not None and data_step in processed_data_index:
            continue
        if not skip_print:
            print(f'skip {data_step}, start from {data_step}')
            skip_print = True
        entry = data_utils.preprocess_model_te_data(args.task_genre, entry)

        cur_verbalizers = [ele.strip() for v in prompt_dict['answer_choices'].values() for ele in v]
        if args.task_genre == 'nli':
            cur_message = prompt_utils.apply_nli_template(entry, prompt_dict['jinja'], prompt_dict['answer_choices'], use_cot)
        elif args.task_genre == 'sa':
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
                res = {
                    'prompt_index': prompt_index,
                    'prompt_info': prompt_dict,
                    'model': args.model,
                    'bit': args.bit,
                    'seed': args.seed,
                    'variant': None,
                    'cot': use_cot,
                    'data_index': data_steps[i],
                    'data_entry': entry_batch[i],
                    'input_promptt': prompt_list[i],
                    'token_info': token_probs[i],
                    'prompt_ppl': prompt_ppl[i],
                }
                if use_cot:
                    res['chat_historyy']: chat_history[i]
                file.write(json.dumps(res) + "\n")
            input_batch = []
            entry_batch = []
            data_steps = []

        if data_step % 100 == 0:
            file.flush()
    file.close()
    print(f'Done: {args.save_path}')


if __name__ == '__main__':
    main()

