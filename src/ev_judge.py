import argparse
import os
REPO_PATH = os.environ['REPO_PATH']
os.chdir(REPO_PATH)

from src.make_requests_to_api_base import generate
from src.utils import get_data_from_dir, setup_logger, cross_battle_join


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setting environment variables for a project')
    
    parser.add_argument(
        '--prompt',
        required=False, 
        default='only_crit',
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
    parser.add_argument(
        '--prepare-instruction',
        required=False,
        default='evaluation_by_criteria',
        type=str,
        help='Set criteria prompt'
    )
    args = parser.parse_args()

    log = setup_logger(log_name=f'{args.prompt}/main_{args.prompt}')

    model_data = get_data_from_dir(
        path=f'{REPO_PATH}/datasets/ru_arena-hard-v0.1/model_answers',
        log=log
    )
    ref_data = get_data_from_dir(
        path=f'{REPO_PATH}/datasets/ru_arena-hard-v0.1/baselines',
        log=log
    )

    model_crit = get_data_from_dir(
        path=f'{REPO_PATH}/artifacts/{args.prepare_instruction}/models',
        log=log
    )
    ref_crit = get_data_from_dir(
        path=f'{REPO_PATH}/artifacts/{args.prepare_instruction}/baselines',
        log=log
    )

    cross_data = cross_battle_join(
        model_data, 
        model_crit,
        ref_data,
        ref_crit
    )

    log.info(f'[main] Start evaluating models for prompt {args.prompt}')
    cross_ev = generate(
        cross_data,
        which_prompt=args.prompt,
        task=f'judge_evaluation_results/{args.prompt}',
        user_mode=args.user,
        num_procs=20,
        judge_model_name='Qwen3-235B-A22B-Instruct-2507',
        battle=True,
        log=setup_logger(log_name=f'{args.prompt}/judge_evaluation'),
    )
