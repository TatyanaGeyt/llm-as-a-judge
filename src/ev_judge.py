import argparse
import os
from src.make_requests_to_api_base import generate
from src.utils import get_data_from_dir, setup_logger, cross_battle_join


REPO_PATH = os.environ['REPO_PATH']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setting environment variables for a project')
    
    parser.add_argument(
        '--prompt',
        required=True, 
        type=str,
        help='prompt given to the Judge'
    )
    args = parser.parse_args()

    log = setup_logger(__name__)

    model_data = get_data_from_dir(
        path=f'{REPO_PATH}/datasets/ru_arena-hard-v0.1/model_answers',
        log=log
    )
    ref_data = get_data_from_dir(
        path=f'{REPO_PATH}/datasets/ru_arena-hard-v0.1/baselines',
        log=log
    )

    model_crit = get_data_from_dir(
        path=f'{REPO_PATH}/evaluation_by_criteria/models',
        log=log
    )
    ref_crit = get_data_from_dir(
        path=f'{REPO_PATH}/evaluation_by_criteria/models',
        log=log
    )

    cross_data = cross_battle_join(
        model_data, 
        model_crit,
        ref_data,
        ref_crit
    )

    cross_ev = generate(
        cross_data,
        which_prompt=args.prompt,
        task=f'judge_evaluation_results{args.prompt}',
        num_procs=20,
        judge_model_name='Qwen3-235B-A22B-Instruct-2507',
        battle=True
    )
