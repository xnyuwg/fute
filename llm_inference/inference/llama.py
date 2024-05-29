import os
import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer, 
    GenerationConfig,
    StoppingCriteria, 
    StoppingCriteriaList,
)
from llm_inference.utils import FileIO


class StoppingCriteriaSub(StoppingCriteria):
    """ Helper class to config stop token for Llama2
    """
    def __init__(self, tokenizer: LlamaTokenizer, stops = [], encounters: int = 1, device: str = 'cuda'):
        super().__init__()
        self.stops = [stop.to(device) for stop in stops]
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if self.tokenizer.decode(stop) == self.tokenizer.decode(last_token):
                return True
        return False


class LlamaInference():
    ACCESS_TOKEN_PATH = './llm_inference/configs/llama_token.key'
    CACHE_DIR = '/scratch/prj/inf_llmcache/hf_cache/'
    file_io = FileIO()
    dirname = os.path.dirname(__file__)

    access_token = open(os.path.expanduser(ACCESS_TOKEN_PATH), 'r').read().strip()

    def __init__(self):
        self.DEVICE = 'cuda'
        self.MODEL_NAME = None
        self.generation_config = None

    def init_model(self, bit: str):
        if bit == '32':
            self.model = LlamaForCausalLM.from_pretrained(
                self.MODEL_NAME,
                device_map="auto",
                token=self.access_token,
                cache_dir=self.CACHE_DIR,
            )
        elif bit == '16':
            self.model = LlamaForCausalLM.from_pretrained(
                self.MODEL_NAME, 
                device_map="auto", 
                token=self.access_token,
                cache_dir=self.CACHE_DIR,
                torch_dtype=torch.float16,
            )
        elif bit == '8':
            self.model = LlamaForCausalLM.from_pretrained(
                self.MODEL_NAME, 
                device_map="auto", 
                token=self.access_token,
                cache_dir=self.CACHE_DIR,
                load_in_8bit=True,
            )
        elif bit == '4':
            self.model = LlamaForCausalLM.from_pretrained(
                self.MODEL_NAME,
                device_map="auto",
                token=self.access_token,
                cache_dir=self.CACHE_DIR,
                load_in_4bit=True,
            )

        self.tokenizer = LlamaTokenizer.from_pretrained(
            self.MODEL_NAME, 
            token=self.access_token,
            cache_dir=self.CACHE_DIR,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _create_stopping_criteria(self, stop_tokens: list):
        stop_token_ids = [self.tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_tokens]
        self.hf_stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(self.tokenizer, stops=stop_token_ids, device=self.DEVICE)])

    def get_verbalizer_scores(self, dist_over_gen_tokens: torch.FloatTensor, verbalizer_ids: list) -> torch.FloatTensor:
        batch_size = dist_over_gen_tokens[0].size(0)
        cur_indicators = [0] * batch_size
        for step_idx, step_scores in enumerate(dist_over_gen_tokens):
            cur_step_tokens = self.tokenizer.batch_decode(step_scores.argmax(dim=-1))

            for idx, ind in enumerate(cur_indicators):
                if ind == 0 and cur_step_tokens[idx].strip():
                    cur_indicators[idx] = step_idx

            if 0 not in cur_indicators:
                break

        dist_over_first_token = torch.stack([dist_over_gen_tokens[ind][j, :] for (j, ind) in enumerate(cur_indicators)])

        verbalizer_score = dist_over_first_token[:, verbalizer_ids]
        out = torch.nan_to_num(verbalizer_score, nan=0.0, posinf=1e5, neginf=-1e5)

        return out.cpu().numpy().tolist()

    @torch.no_grad()
    def _generate(
        self, prompt, 
        config: dict, 
        stop_tokens: list,
        return_dict: bool = True,
        output_scores: bool = True,
    ):

        if isinstance(prompt[0], str):
            model_inputs = self.tokenizer.batch_encode_plus(
                prompt, 
                padding='longest',
                return_tensors="pt", 
                add_special_tokens=False
            )['input_ids'].to(self.DEVICE)

        else:
            if len(prompt) > 1:
                model_inputs = self.patch_input_batch(prompt).to(self.DEVICE)
            else:
                model_inputs = prompt[0].to(self.DEVICE)

        if stop_tokens:
            self._create_stopping_criteria(stop_tokens)

        if config:
            self._set_generation_config(config)

            if stop_tokens:
                output = self.model.generate(
                    model_inputs.long(),
                    generation_config=self.generation_config,
                    stopping_criteria=self.hf_stopping_criteria,
                    return_dict_in_generate=return_dict, 
                    output_scores=output_scores,
                )
            else:
                output = self.model.generate(
                    model_inputs.long(),
                    generation_config=self.generation_config,
                    return_dict_in_generate=return_dict, 
                    output_scores=output_scores,
                )
        else:
            if stop_tokens:
                output = self.model.generate(
                    model_inputs.long(),
                    stopping_criteria=self.hf_stopping_criteria,
                    return_dict_in_generate=return_dict, 
                    output_scores=output_scores,
                )

            else:
                output = self.model.generate(
                    model_inputs.long(),
                    return_dict_in_generate=return_dict, 
                    output_scores=output_scores,
                )

        return output, model_inputs

    @torch.no_grad()
    def inference(
        self, 
        prompt: str, 
        verbalizers: list, 
        config: dict = {}, 
        cot: bool = False,
        stop_tokens: list = [],
        new_instruction=None,
        second_round_config=None,
    ) -> tuple:

        self.batch_size = len(prompt)

        if cot:
            output, model_inputs = self._generate(prompt, config, stop_tokens, return_dict=False, output_scores=False)

            new_instruction = '[INST] ' + new_instruction.replace('{{label_space}}', ', '.join(verbalizers)) + ' [/INST]'

            second_round = torch.cat((
                torch.tensor([[self.tokenizer.bos_token_id]]),
                self.tokenizer.encode(new_instruction, add_special_tokens=False, return_tensors='pt')
            ), dim=-1).to(self.DEVICE)

            second_round_input = [torch.cat((output, second_round.expand(self.batch_size, -1)), dim=-1)]

            output, complete_chat = self._generate(second_round_input, second_round_config, stop_tokens)

        else:
            output, model_inputs = self._generate(prompt, config, stop_tokens)

        verbalizer_ids = [self.tokenizer.encode(verbalizer, add_special_tokens=False) for verbalizer in verbalizers]
        for i, vid in enumerate(verbalizer_ids):
            if len(vid) != 1:
                print(f"error:  len(vid) != 1, with v={verbalizers[i]} and vid={vid}")
        verbalizer_ids = [v[0] for v in verbalizer_ids if len(v) == 1]

        dist_over_gen_tokens = output.scores

        score_over_verbalizer_words = self.get_verbalizer_scores(dist_over_gen_tokens, verbalizer_ids)

        dist_out = []
        for s_word_list in score_over_verbalizer_words:

            assert len(s_word_list) == len(verbalizers), "The number of verbalizers and the number of word scores are not equal."

            dist_out.append({
                'word_scores': s_word_list,
                'verbalizers': verbalizers,
            })

        text_prompt_list = []
        text_prompt_list_for_ppl = []
        for input in model_inputs:
            text_prompt_list.append(self.tokenizer.decode(input.squeeze(), skip_special_tokens=False))
            text_prompt_list_for_ppl.append(self.tokenizer.decode(input.squeeze(), skip_special_tokens=False))

        complete_chat_list = []
        if cot:
            for input in complete_chat:
                complete_chat_list.append(self.tokenizer.decode(input.squeeze(), skip_special_tokens=False))

        ppl_result = self.compute_ppl(text_prompt_list_for_ppl)

        output = [self.tokenizer.decode(o, skip_special_tokens=False) for o in output.sequences]

        return output, dist_out, ppl_result, text_prompt_list, complete_chat_list

    @staticmethod
    def format_prompt(chatgpt_prompt: list, model_param: str = '7b') -> torch.Tensor:
        access_token = open(os.path.expanduser(LlamaInference.ACCESS_TOKEN_PATH), 'r').read().strip()

        tokenizer = LlamaTokenizer.from_pretrained(
            f'meta-llama/Llama-2-{model_param}-chat-hf',
            token=access_token,
            cache_dir='/scratch/prj/inf_llmcache/hf_cache/',
        )

        BOS_TOKEN, EOS_TOKEN = tokenizer.bos_token_id, tokenizer.eos_token_id
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        llama_prompt = tokenizer.encode(
            f"{B_INST} {B_SYS} {chatgpt_prompt[0]['content'].strip()} {E_SYS}",
            add_special_tokens=False,
            return_tensors='pt'
        )

        llama_prompt = torch.concat((torch.tensor([[BOS_TOKEN]]), llama_prompt), dim=-1)

        for idx, content_dict in enumerate(chatgpt_prompt):

            if idx == 0:
                continue

            elif idx == 1:
                llama_prompt = torch.concat((
                    llama_prompt, 
                    tokenizer.encode(
                        f"{content_dict['content']} {E_INST}",
                        add_special_tokens=False,
                        return_tensors='pt'
                    )
                ), dim=-1)

            else:
                if content_dict['role'] == 'user':
                    temp_prompt = tokenizer.encode(
                        f"{B_INST} {content_dict['content']} {E_INST}",
                        add_special_tokens=False,
                        return_tensors='pt',
                    )
                    llama_prompt = torch.concat((llama_prompt, torch.tensor([[BOS_TOKEN]]), temp_prompt), dim=-1)

                elif content_dict['role'] == 'assistant':
                    temp_prompt = tokenizer.encode(
                        f"{content_dict['content']}",
                        add_special_tokens=False,
                        return_tensors='pt',
                    )
                    llama_prompt = torch.concat((llama_prompt, temp_prompt, torch.tensor([[EOS_TOKEN]])), dim=-1)

        return llama_prompt

    def patch_input_batch(self, input_batch: list) -> torch.Tensor:
        max_len = max([ele.size(-1) for ele in input_batch])
        # apply padding to max sequence length
        pad_token_id = self.tokenizer.pad_token_id
        input_batch = [torch.cat((torch.ones(max_len - ele.size(-1)) * pad_token_id, ele.squeeze())) for ele in input_batch]
        return torch.stack(input_batch)

    def compute_ppl(
        self, 
        prompts: list, 
        batch_size: int = 16, 
        add_start_token: bool = True, 
        max_length=None
        ):

        if self.tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(self.tokenizer.special_tokens_map_extended.values())
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            self.tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        if add_start_token and max_length:
            assert (
                self.tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_length - 1
        else:
            max_tokenized_len = max_length

        encodings = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            truncation=True if max_tokenized_len else False,
            max_length=max_tokenized_len,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.DEVICE)

        encoded_texts = encodings["input_ids"]
        attn_masks = encodings["attention_mask"]

        if add_start_token:
            assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
        else:
            assert torch.all(
                torch.ge(attn_masks.sum(1), 2)
            ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

        ppls = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in range(0, len(encoded_texts), batch_size):
            end_index = min(start_index + batch_size, len(encoded_texts))
            encoded_batch = encoded_texts[start_index:end_index]
            attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[self.tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(self.DEVICE)
                encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                attn_mask = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(self.DEVICE), attn_mask], dim=1
                )

            labels = encoded_batch

            with torch.no_grad():
                out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

        return ppls

    def set_model(self, model_name: str, chat: bool = True):
        if '-' in model_name:
            model_name = model_name.split('-')[-1]

        if chat:
            self.MODEL_NAME = f'meta-llama/Llama-2-{model_name}-chat-hf'
        else:
            self.MODEL_NAME = f'meta-llama/Llama-2-{model_name}-hf'

    def _set_generation_config(self, config: dict):
        generation_config = GenerationConfig.from_pretrained(
            self.MODEL_NAME,
            cache_dir=self.CACHE_DIR,
            use_auth_token=self.access_token,
            **config,
        )
        self.generation_config = generation_config
