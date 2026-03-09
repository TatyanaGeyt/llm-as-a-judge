from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import json
import string
import requests
from datetime import datetime
from benchmark.tools import (
    get_json_files,
    get_data,
    get_coords,
    setup_logger,
    write_json,
    make_battle,
    check_requests
)


class ApiBase:
    def __init__(self, host, secret_key):
        self.host = host
        self.key = secret_key
    
    def make_request(self, data):
        r = requests.post(
            f'{self.host}/v1/chat/completions',
            json=data,
            headers={'Authorization': 'Bearer ' + self.key},
            timeout=60
        )
        if r.status_code != 200:
            return {
                'error': r.status_code,
                'details': r.text
            }
        r = r.json()['choices'][0]['message']['content']
        return {
            'output': r
        }

class Prompt:
    _prompt_counter = 0

    def __init__(self,
        prompt_name: str='',
        prompt_pattern: str='',
        example_file: str='prompts.json'
    ):
        self.set_prompt(prompt_pattern, prompt_name)
        self.file = example_file

    def set_prompt(self,
        prompt_pattern: str,
        name: str=''
    ):
        self.name = self._generate_name(name)
        self.pattern = prompt_pattern
        self.set = bool(prompt_pattern)

    def save_prompt(self):
        """Saves current prompt to file, maintaining the dictionary structure."""
        required_args = [
            fname for _, fname, _, _ in string.Formatter().parse(self.pattern) if fname
        ]

        data = {}
        if os.path.exists(self.file) and os.path.getsize(self.file) > 0:
            with open(self.file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}

        data[self.name] = {
            'pattern': self.pattern,
            'args_list': required_args
        }

        with open(self.file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"Prompt '{self.name}' saved to {self.file}")

    def load_prompt(self, prompt_name: str):
        """Loads pattern from file and applies new args."""
        if not os.path.exists(self.file):
            raise FileNotFoundError(f"File {self.file} not found.")

        with open(self.file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if prompt_name not in data:
            raise KeyError(f"Prompt '{prompt_name}' not found in library.")

        entry = data[prompt_name]
        self.name = prompt_name
        self.pattern = entry['pattern']
        print(f"Prompt '{prompt_name}' loaded successfully.")
    
    def apply_pattern(self, args):
        if not self.pattern:
            return ""
        try:
            return self.pattern.format(**args)
        except KeyError as e:
            raise KeyError(f"Error formatting prompt: Missing argument {e}")
    
    def _generate_name(self, prompt_name):
        if prompt_name:
            return prompt_name
        else:
            name = f'userprompt_{self._prompt_counter}'
            self._prompt_counter += 1
            return name


class ModelInstance:
    def __init__(self, 
        model_name: str,
        answers_file: str,
        role: str # 'baseline' or 'ev_model'
    ):
        self.name = model_name
        self.path = Path(answers_file) \
            if isinstance(answers_file, str) \
            else answers_file
        self.data = get_data(self.path)
        self.role = role
    
    def set_data(self, data):
        self.data = data
    
    def set_role(self, role):
        self.role = role

    def split_batches(self, batch_size):
        if not self.data or batch_size <= 0:
            return
        new_data = [
            self.data[i : i + batch_size] 
            for i in range(0, len(self.data), batch_size)
        ]
        self.data = new_data


class ModelList:
    def __init__(self, path: str='', baselines: list[str]=[]):
        self.path = path
        self.baselines_list = baselines
        self.ev_models_list = []

        self.baselines = []
        self.ev_models = []

        self.load_data()
        
    def set(self, path, baselines):
        self.path = path
        self.baselines_list = baselines
        self.load_data()
    
    def load_data(self):
        self.baselines = []
        self.ev_models = []
        if not self.path:
            self.models = {}
            return
        files = get_json_files(self.path)
        self.models = self._set_models(files, self.baselines_list)

    def _set_models(self, files, baselines):
        models = {}
        for file in files:
            model_name = file.stem
            models[model_name] = ModelInstance(model_name, file, role='ev_model')
        ev_models = []
        for m in models.keys():
            if m in baselines:
                models[m].set_role('baseline')
                self.baselines.append(models[m])
            else:
                ev_models.append(m)
                self.ev_models.append(models[m])
        self.ev_models_list = ev_models
        return models


class BaseJudge:
    def __init__(self):
        pass


class LLMAsJudge(BaseJudge):
    def __init__(self, 
            api_base_host: str, 
            openai_api_key: str, 
            style_control: bool=True,
            num_procs=20,
            judge_model: str='Qwen3-235B-A22B-Instruct-2507',
            model_config: dict={},
            prompt_pattern: str='',
            dataset_path: str='',
            baselines: list[str]=[],
            log_dir: str=''
        ):
        super().__init__()
        self.api_base = ApiBase(api_base_host, openai_api_key)
        self.judge_model = judge_model
        self.config = model_config
        self.style_control = style_control
        self.num_procs = num_procs
        self.prompt = Prompt(prompt_pattern)
        self.instruction_system = 'You are a meticulous and impartial assistant designed to evaluate the quality of language model responses. Your task is to follow a strict, step-by-step evaluation procedure to respond.'
        self.models = ModelList(dataset_path, baselines)
        self.log_dir = log_dir
        self.log = None
    
    def set_prompt(self, prompt_pattern):
        self.prompt.set_prompt(prompt_pattern)

    def set_models(self, path, baselines):
        self.models.set(path, baselines)

    def set_config(self,
        max_tokens: str=256,
        temperature: float=0.0,
        top_p:  float=0.9,
        top_k: int=40,
        repetition_penalty: float=1.0,
        stop=None,
        stop_token_ids=None,
        n=1,
        add_generation_prompt=True,
        skip_special_tokens=True,
        continue_final_message=False,
        include_stop_str_in_output=False,
        chat_template_kwargs={'enable_thinking': False},
        config: dict={}
    ):
        if config:
            if 'model' not in config.keys():
                config['model'] = self.judge_model
            self.config = config
            return
        self.config = {
            'model': self.judge_model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p':  top_p,
            'top_k': top_k,
            'repetition_penalty': repetition_penalty,
            'stop': stop,
            'stop_token_ids': stop_token_ids,
            'n': n,
            'add_generation_prompt': add_generation_prompt,
            'skip_special_tokens': skip_special_tokens,
            'continue_final_message': continue_final_message,
            'include_stop_str_in_output': include_stop_str_in_output,
            'chat_template_kwargs': chat_template_kwargs
        }
    
    def create_message(self, sample):
        msg = [
            {'role': 'system', 'content': self.instruction_system},
            {'role': 'user', 'content': self.prompt.apply_pattern(sample)},
        ]
        return {
            'messages': msg,
            **self.config
        }

    def generate_batch(self, samples_batch, batch_id):
        messages = [self.create_message(sample) for sample in samples_batch]
        coords = [get_coords(sample) for sample in samples_batch]
        results_ordered = [None] * len(samples_batch)
        with ThreadPoolExecutor(max_workers=self.num_procs) as executor:
            future_to_idx = {
                executor.submit(self.api_base.make_request, msg): (i, coords[i])
                for i, msg in enumerate(messages)
            }
            pbar = tqdm(
                as_completed(future_to_idx),
                total=len(samples_batch),
                desc="Generating Batches"
            )
            for future in pbar:
                idx, coord = future_to_idx[future]
                try:
                    result = future.result()
                    if 'error' in result.keys():
                        raise ValueError(f'request_error CODE {result["error"]}: {result["details"]}')
                    results_ordered[idx] = {**coord, **result}
                    self.log.info(f'[{idx}] ok')
                except Exception as e:
                    self.log.error(f"Error in task {idx}: {e}")
                    results_ordered[idx] = {
                        'id': idx,
                        'error': str(e)
                    }
        received = list(results_ordered)
        return received

    def evaluate(
        self,
        single_model='',
        models=[],
        cross_battle=False,
        load_path: str=''
    ):
        assert self.prompt.set == True, 'Prompt for Judge is not set!'
        if single_model:
            assert not models, "`single model` and `models` modes can not me set simultaneously!"
            models = [single_model]
        if not models:
            models = self.models.ev_models_list

        ev_models = [m for m in self.models.ev_models if m.name in models]
        run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log = setup_logger(self.log_dir, log_name=f'run_{run_time}')

        result = {}
        load = None
        if load_path:
            load = Path(load_path) / f'{self.prompt.name}'
            load.mkdir(parents=True, exist_ok=True)
    
        batch_id = 0
        for m in ev_models:
            if cross_battle:
                data = {}
                for b in self.models.baselines:
                    battle = make_battle(m.data, b.data)
                    received = self.generate_batch(battle, batch_id)
                    data[b.name]  = check_requests(battle, received, batch_id, self)
                    
                    batch_id += 1
            else:
                received = self.generate_batch(m.data, batch_id)
                data = check_requests(m.data, received, batch_id, self)
                batch_id += 1

            if load_path:
                file = load / f'{m.name}.json'
                write_json(file, data, self.log)
            result[m.name] = data
        
        if load_path:
            file = load / f'total.json'
            write_json(file, result, self.log)
        
        return result