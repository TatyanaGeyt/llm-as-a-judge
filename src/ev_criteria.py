import argparse
from pathlib import Path
import os
REPO_PATH = os.environ['REPO_PATH']
os.chdir(REPO_PATH)

from src.make_requests_to_api_base import generate
from src.utils import get_data_from_dir, setup_logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setting environment variables for a project')
    
    parser.add_argument(
        '--prompt',
        required=False, 
        default='default_criteria_prompt',
        type=str,
        help='prompt given to the Judge'
    )
    parser.add_argument(
        '--user',
        required=False, 
        action='store_true',
        default=False,
        help='Enable user mode'
    )
    args = parser.parse_args()

    log = setup_logger(log_name=f'{args.prompt}/main_{args.prompt}')

    model_data = get_data_from_dir(
        path=f'{REPO_PATH}/datasets/ru_arena-hard-v0.1/model_answers',
        log=log
    )
    log.info(f'[main] Loaded data for models')

    ref_data = get_data_from_dir(
        path=f'{REPO_PATH}/datasets/ru_arena-hard-v0.1/baselines',
        log=log
    )
    log.info(f'[main] Loaded data for baselines')

    res_dir = Path(REPO_PATH) / f'artifacts/{args.prompt}/models'
    res_dir.mkdir(parents=True, exist_ok=True)
    res_dir = Path(REPO_PATH) / f'artifacts/{args.prompt}/baselines'
    res_dir.mkdir(parents=True, exist_ok=True)
    log.info(f'[main] Directory was created')

    log.info(f'[main] Extracting criterias for models for prompt {args.prompt}')
    model_ev = generate(
        model_data,
        which_prompt=args.prompt,
        task=f'{args.prompt}/models',
        user_mode=args.user,
        log=setup_logger(log_name=f'{args.prompt}/models/evaluate_models'),
        num_procs=20,
        judge_model_name='Qwen3-235B-A22B-Instruct-2507',
        battle=False
    )

    log.info(f'[main] Extracting criterias for baselines for prompt {args.prompt}')
    rev_ev = generate(
        ref_data,
        which_prompt=args.prompt,
        task=f'{args.prompt}/baselines',
        user_mode=args.user,
        log=setup_logger(log_name=f'{args.prompt}/baselines/evaluate_baselines'),
        num_procs=20,
        judge_model_name='Qwen3-235B-A22B-Instruct-2507',
        battle=False
    )