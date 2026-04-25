from concurrent.futures import ThreadPoolExecutor, as_completed
from math import log
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
import os
import json
from datetime import datetime
from benchmark.tools import (
    get_coords,
    setup_logger,
    check_requests,
    extract_format_args,
    split_batches,
    swap_example,
)
from benchmark.style_control import (
    apply_style_control_state,
    compute_style_control,
    fit_length_control_state,
    fit_style_control_state,
    get_element_counts,
    get_length_control_counts,
    scale_to_elo,
    winner_labels_to_outcomes,
)


class ApiBase:
    def __init__(self, host, secret_key, path=''):
        self.host = host
        self.key = secret_key
        self.path = path
    
    def make_request(self, data, path=''):
        if not path:
            path = self.path
        r = requests.post(
            f'{self.host}/{path}',
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

    def __repr__(self):
        return f'host: {self.host}'

class Prompt:
    _prompt_counter = 0

    def __init__(self,
        prompt_name: str='',
        prompt_pattern: str='',
        example_file: str='artifacts/prompts/prompts.json'
    ):
        self.set_prompt(prompt_pattern, prompt_name)
        self.file = example_file

    def set_prompt(self,
        prompt_pattern: str,
        name: str='',
        field_map: dict=None
    ):
        self.name = self._generate_name(name)
        self.pattern = prompt_pattern
        self.set = bool(prompt_pattern)
        self.required_args = extract_format_args(self.pattern)
        self.field_map = field_map or {}
        

    def save_prompt(self, file: str=''):
        """Saves current prompt to file, maintaining the dictionary structure."""
        if not file:
            file = self.file
        data = {}
        if os.path.exists(file) and os.path.getsize(file) > 0:
            with open(file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
        data[self.name] = self.pattern

        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Prompt '{self.name}' saved to {self.file}")

    def load_prompt(self, prompt_name: str, prompt_file: str=''):
        """Loads pattern from file"""
        if not prompt_file:
            prompt_file = self.file
            if not os.path.exists(prompt_file):
                raise FileNotFoundError(f"File {prompt_file} not found.")

        with open(prompt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if prompt_name not in data:
            raise KeyError(f"Prompt '{prompt_name}' not found in library.")

        self.name = prompt_name
        self.pattern = data[prompt_name]
        self.set = bool(self.pattern)
        self.required_args = extract_format_args(self.pattern)
        print(f"Prompt '{prompt_name}' loaded successfully.")
    
    def set_mapping(self, mapping):
        self.field_map = mapping
    
    def apply_pattern(self, args):
        if not self.pattern:
            return ""
        try:
            mapped_args = args
            if self.field_map:
                mapped_args = {}
                for key in self.required_args:
                    source_key = self.field_map.get(key, key)
                    if source_key not in args:
                        raise KeyError(
                            f"Missing argument '{key}' (mapped from '{source_key}')"
                        )
                    mapped_args[key] = args[source_key]
            return self.pattern.format(**mapped_args)
        except KeyError as e:
            print(args, mapped_args)
            raise KeyError(f"Error formatting prompt: Missing argument {e}, required arguments: {self.required_args}. Use LLMAsJudge.dataset_mapping(...)")
    
    def _generate_name(self, prompt_name):
        if prompt_name:
            return prompt_name
        else:
            name = f'userprompt_{self._prompt_counter}'
            self._prompt_counter += 1
            return name
    
    def __repr__(self):
        return f'prompt: {self.name}'


class BaseJudge:
    def __init__(self):
        pass


class LLMAsJudge(BaseJudge):
    def __init__(self, 
            api_base_host: str, 
            openai_api_key: str,
            num_procs=20,
            judge_model: str='',
            judge_path: str='',
            model_config: dict={},
            prompt_pattern: str='',
            log_dir: str='logs'
        ):
        super().__init__()
        self.api_base = ApiBase(api_base_host, openai_api_key)
        self.judge_model = judge_model
        self.judge_path = judge_path
        self.config = model_config
        self.num_procs = num_procs
        self.prompt = Prompt(prompt_pattern)
        self.instruction_system = 'You are a meticulous and impartial assistant designed to evaluate the quality of language model responses. Your task is to follow a strict, step-by-step evaluation procedure to respond.'
        self.log_dir = log_dir
        self.log = None
        self._style_bt_state: Optional[dict] = None
        self._length_bt_state: Optional[dict] = None

    def set_prompt(self, prompt_pattern='', prompt_name='', from_file='', mapping=None):
        if from_file:
            self.prompt.load_prompt(prompt_name=prompt_name, prompt_file=from_file)
        elif prompt_pattern:
            self.prompt.set_prompt(prompt_pattern, name=prompt_name)
        else:
            raise ValueError(f'[set_prompt] ERROR - wrong arguments')
        self.dataset_mapping(mapping)
    
    def dataset_mapping(self, mapping):
        self.prompt.set_mapping(mapping)

    def set_config(self,
        config: dict={},
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
        chat_template_kwargs={'enable_thinking': False}
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
                executor.submit(self.api_base.make_request, msg, self.judge_path): (i, coords[i])
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
    
    def evaluate(self, data, batch_id, swapping_pos=True):
        recieved = check_requests(data, self.generate_batch(data, batch_id), batch_id, self)

        if swapping_pos:
            data_forward = [swap_example(d) for d in data]
            print('Swapping positions: start')
            recieved_swapped = check_requests(data, self.generate_batch(data_forward, batch_id), batch_id, self)

            return [
                {'orig': orig, 'swapped': swapped}
                for (orig, swapped) in zip(recieved, recieved_swapped)
            ]
    
        return recieved

    def evaluate_dataset(self, dataset: list[dict], swapping_pos=True):
        """Run the judge on ``dataset`` in batches; optionally run a second pass with swapped A/B order."""
        assert self.prompt.set is True, 'Prompt for Judge is not set!'
        run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log = setup_logger(self.log_dir, log_name=f'run_{run_time}')
        data = split_batches(dataset)
        
        output = []
        print(f'{len(data)} batches')
        for id, batch in enumerate(data):
            print(f'Batch {id} started')
            try:
                output.append(self.evaluate(batch, id, swapping_pos))
            except Exception as e:
                print(f'Batch {id} - error: {e}')
                output.append([])
        return [item for sublist in output for item in sublist]
    
    def style_control(self, data, winner):
        print('OK')
        df = {'model_a': [], 'model_b': [], 'model_a_style': [], 'model_b_style': []}

        for d in data:
            df['model_a'].append(d['model'])
            df['model_b'].append(d['reference'])
            df['model_a_style'].append(get_element_counts(d['model_answer']))
            df['model_b_style'].append(get_element_counts(d['reference_answer']))

        df = pd.DataFrame(df)
        df['winner'] = winner

        ratings, models = compute_style_control(df)
        return scale_to_elo(ratings, models), models

    def train_style_control(
        self,
        train_dataset: List[dict],
        winners: List[Union[int, float, str, bool]],
        *,
        alpha: Optional[float] = None,
        reg: float = 10.0,
        tol: float = 1e-6,
    ) -> dict:
        """
        Fit FastChat contextual BT + style weights on ``train_dataset``.

        Each row must include ``model``, ``reference``, ``model_answer``, ``reference_answer``.
        ``winners`` is parallel to rows: ``1`` / ``\"model\"`` / ``\"A\"`` = side A (``model``) won,
        ``0`` / ``\"reference\"`` / ``\"B\"`` = side B won.

        Stores internal state for :meth:`apply_style_control` and returns the same dict
        (``models``, ``params``, ``style_mean``, ``style_std``, ``alpha``, …).
        """
        if not train_dataset:
            raise ValueError("train_dataset is empty")
        y = winner_labels_to_outcomes(winners, len(train_dataset))
        rows = {
            "model_a": [],
            "model_b": [],
            "model_a_style": [],
            "model_b_style": [],
            "winner": y,
        }
        for d in train_dataset:
            rows["model_a"].append(d["model"])
            rows["model_b"].append(d["reference"])
            rows["model_a_style"].append(get_element_counts(d["model_answer"]))
            rows["model_b_style"].append(get_element_counts(d["reference_answer"]))
        df = pd.DataFrame(rows)
        a_used = float(alpha) if alpha is not None else log(10.0)
        self._style_bt_state = fit_style_control_state(df, alpha=a_used, reg=reg, tol=tol)
        return self._style_bt_state

    def apply_style_control(
        self,
        sample: dict,
        judge_label: Any,
        *,
        judge_strength: float = 1.0,
    ) -> float:
        """
        For one battle dict (same keys as training) and a judge verdict, return a probability in
        ``(0, 1)`` that side **A** (``sample['model']``) wins, using stored style-control state:

        ``sigmoid(alpha*(r_A-r_B) + w·f_norm + judge_strength * sign(judge_label))``.

        Call :meth:`train_style_control` first. Unknown model names use rating ``0``.
        """
        if self._style_bt_state is None:
            raise RuntimeError("No fitted style control; call train_style_control(...) first.")
        ca = get_element_counts(sample["model_answer"])
        cb = get_element_counts(sample["reference_answer"])
        return apply_style_control_state(
            self._style_bt_state,
            model_a_name=sample["model"],
            model_b_name=sample["reference"],
            counts_a=ca,
            counts_b=cb,
            judge_label=judge_label,
            judge_strength=judge_strength,
        )

    def clear_style_control(self) -> None:
        """Drop state from :meth:`train_style_control`."""
        self._style_bt_state = None

    def train_length_control(
        self,
        train_dataset: List[dict],
        winners: List[Union[int, float, str, bool]],
        *,
        alpha: Optional[float] = None,
        reg: float = 10.0,
        tol: float = 1e-6,
    ) -> dict:
        """
        Fit contextual BT with a **single** contextual feature: relative answer length
        (same pipeline as style control, :data:`LENGTH_CONTROL_ELEMENTS`).

        Rows need ``model``, ``reference``, ``model_answer``, ``reference_answer``.
        ``winners`` is encoded like :meth:`train_style_control`.

        Stores state for :meth:`apply_length_control` (independent of style-control state).
        """
        if not train_dataset:
            raise ValueError("train_dataset is empty")
        y = winner_labels_to_outcomes(winners, len(train_dataset))
        rows = {
            "model_a": [],
            "model_b": [],
            "model_a_style": [],
            "model_b_style": [],
            "winner": y,
        }
        for d in train_dataset:
            rows["model_a"].append(d["model"])
            rows["model_b"].append(d["reference"])
            rows["model_a_style"].append(get_length_control_counts(d["model_answer"]))
            rows["model_b_style"].append(get_length_control_counts(d["reference_answer"]))
        df = pd.DataFrame(rows)
        a_used = float(alpha) if alpha is not None else log(10.0)
        self._length_bt_state = fit_length_control_state(df, alpha=a_used, reg=reg, tol=tol)
        return self._length_bt_state

    def apply_length_control(
        self,
        sample: dict,
        judge_label: Any,
        *,
        judge_strength: float = 1.0,
    ) -> float:
        """
        P(side ``sample['model']`` wins) using **length-only** fitted state
        (see :meth:`train_length_control`). Same logit shape as :meth:`apply_style_control`.
        """
        if self._length_bt_state is None:
            raise RuntimeError("No fitted length control; call train_length_control(...) first.")
        ca = get_length_control_counts(sample["model_answer"])
        cb = get_length_control_counts(sample["reference_answer"])
        return apply_style_control_state(
            self._length_bt_state,
            model_a_name=sample["model"],
            model_b_name=sample["reference"],
            counts_a=ca,
            counts_b=cb,
            judge_label=judge_label,
            judge_strength=judge_strength,
        )

    def clear_length_control(self) -> None:
        """Drop state from :meth:`train_length_control`."""
        self._length_bt_state = None

    def __repr__(self):
        return json.dumps(self.config, indent=4)