import requests
import os
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from artifacts.prompts import prompts
from src.utils import get_coords, setup_logger, write_json


REPO_PATH = os.environ['REPO_PATH']

def create_message(sample: Dict, which_prompt: str):
    instruction_system = 'You are a meticulous and impartial assistant designed to evaluate the quality of language model responses. Your task is to follow a strict, step-by-step evaluation procedure to respond.'
    user_content = getattr(prompts, which_prompt)(sample)
    return [
        {'role': 'system', 'content': instruction_system},
        {'role': 'user', 'content': user_content},
    ]


def make_request(sample, judge_model_name):
    
    msg = create_message(sample)
    r = requests.post(
        f'{os.environ['API_BASE']}/v1/chat/completions',
        json={
            'messages': msg,
            'model': judge_model_name,
            'max_tokens': 128,
            'temperature': 0.0,
            'top_p':  0.9,
            'top_k': 40,
            'repetition_penalty': 1.0,
            'stop': None,
            'stop_token_ids': None,
            'n': 1,
            'add_generation_prompt': True,
            'skip_special_tokens': True,
            'continue_final_message': False,
            'include_stop_str_in_output': False,
            'chat_template_kwargs': {'enable_thinking': False}
        },
        headers={'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY']}
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


def generate_batch(
    samples_batch,
    num_procs,
    judge_model_name,
    log,
    which_prompt
):
    results_ordered = [None] * len(samples_batch)
    with ThreadPoolExecutor(max_workers=num_procs) as executor:
        future_to_idx = {
            executor.submit(make_request, sample, judge_model_name, which_prompt): (i, get_coords(sample))
            for i, sample in enumerate(samples_batch)
        }
        pbar = tqdm(
            as_completed(future_to_idx),
            total=len(samples_batch),
            desc="Generating Batches"
        )
        for future in pbar:
            idx, coords = future_to_idx[future]
            try:
                result = future.result()
                if 'error' in result.keys():
                    raise ValueError(f'request_error CODE {result["error"]}: {result['details']}')
                results_ordered[idx] = coords | result
                log.info(f'[{idx}] ok')
            except Exception as e:
                log.error(f"Error in task {idx}: {e}", 'ERROR')
                results_ordered[idx] = {
                    'id': idx,
                    'error': str(e)
                }
    return list(results_ordered)


def generate_for_model(
    model,
    answ_data,
    which_prompt,
    task,
    num_procs=20,
    judge_model_name='Qwen3-235B-A22B-Instruct-2507',
    log=None,
    battle=False
):
    log_dir = Path(REPO_PATH) / f'logs/{task}'
    log_dir.mkdir(exist_ok=True)

    if battle:
        results = {}
        for r in answ_data: # против каждого baseline (r)
            if not log:
                log = setup_logger(f'{task}/{model}_{r}.log')
                log.info(f'Start')

            res = generate_batch(answ_data[r], num_procs, judge_model_name, log, which_prompt)
            res = check_requests(
                res,
                answ_data,
                task,
                which_prompt,
                num_procs,
                judge_model_name
            )

            res_dir = Path(REPO_PATH) / f'artifacts/{task}'
            res_dir.mkdir(exist_ok=True)
            write_json(f'{REPO_PATH}/artifacts/{task}/{model}_{r}.json', res, log)
            results[r] = res
    else:
        if not log:
            log = setup_logger(f'{task}/{model}.log')
            log.info(f'Start')

        res = generate_batch(answ_data, num_procs, judge_model_name, log, which_prompt)
        res = check_requests(
            res,
            answ_data,
            task,
            which_prompt,
            num_procs,
            judge_model_name
        )

        res_dir = Path(REPO_PATH) / f'artifacts/{task}'
        res_dir.mkdir(exist_ok=True)
        write_json(f'{REPO_PATH}/artifacts/{task}/{model}.json', res, log)
        results = res

    return results


def generate(
    data,
    which_prompt,
    task,
    num_procs=20,
    judge_model_name='Qwen3-235B-A22B-Instruct-2507',
    battle=False
):
    results = {}
    log = setup_logger(f'{task}/total.log')
    
    for m in data:
        try:
            res = generate_for_model(
                model=m,
                answ_data=data[m],
                which_prompt=which_prompt,
                task=task,
                num_procs=num_procs,
                judge_model_name=judge_model_name,
                battle=battle
            )
            log.info(f'Successfully got res for model {m}')
            results[m] = res
        except Exception as e:
            log.error(f'{e}')

    write_json(f'{REPO_PATH}/artifacts/{task}/total_criteria.json', results, log)
    return results


def check_requests(
    check_data, 
    source_data,
    task,
    which_prompt,
    num_procs,
    judge_model_name
):
    log = setup_logger(f'{task}/restart.log')

    restart = True
    iter = 0
    while restart:
        log.info(f'[RESTART] iteration {iter}')
        restart = False
        for m in check_data:
            good_tasks = [
                d for d in check_data[m]
                if 'error' not in d.keys()
            ]
            restart_id = [
                d['id'] for d in check_data[m]
                if 'error' in d.keys()
            ]
            if not restart_id:
                continue

            restart = True
            log.info(f'[RESTART] model {m}: {restart_id}')
            
            restart_data = [
                d for d in source_data[m]
                if d['id'] in restart_id
            ]
            good_tasks += generate_batch(
                samples_batch=restart_data,
                num_procs=num_procs,
                judge_model_name=judge_model_name,
                log=log,
                which_prompt=which_prompt
            )
            check_data[m] = sorted(good_tasks, key=lambda x: x['id'])
            iter += 1
    return check_data