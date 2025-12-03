from src.make_requests_to_api_base import generate
from src.utils import get_data_for_criteria, setup_logger
import os


REPO_PATH = os.environ['REPO_PATH']

log = setup_logger(__name__)

model_data = get_data_for_criteria(
    path=f'{REPO_PATH}/datasets/ru_arena-hard-v0.1/model_answers',
    log=log
)
ref_data = get_data_for_criteria(
    path=f'{REPO_PATH}/datasets/ru_arena-hard-v0.1/baselines',
    log=log
)

model_ev = generate(
    model_data,
    which_prompt='default_criteria_prompt',
    task='evaluation_by_criteria/models',
    num_procs=20,
    judge_model_name='Qwen3-235B-A22B-Instruct-2507',
    battle=False
)

rev_ev = generate(
    ref_data,
    which_prompt='default_criteria_prompt',
    task='evaluation_by_criteria/baselines',
    num_procs=20,
    judge_model_name='Qwen3-235B-A22B-Instruct-2507',
    battle=False
)