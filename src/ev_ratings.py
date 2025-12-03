import pandas as pd
from src.calculate_ratings import compute_elo_ratings, winner
from pathlib import Path
import os


REPO_PATH = os.environ['REPO_PATH']

answs_dir = Path(f"{REPO_PATH}/artifacts/judge_evaluation_results")
"/model_answers/reference"

baseline_models = [
    'gpt-4-1106-preview',
    'gpt-4o-mini',
    'Qwen2.5-32B-Instruct',
    'gigachat_max_26.20_uncen'
 ]

models = [f.stem for f in answs_dir.glob("*.json")]
assert models

to_elo = pd.DataFrame(columns=['model', 'model_ref', 'winner'])
for m in models:
    cur_res = models[m]
    df = pd.DataFrame()
    df['model_ref'] = [d['r'] for d in cur_res]
    df['model'] = [d['m'] for d in cur_res]
    df['winner'] = [winner(d) for d in cur_res]
    to_elo = pd.concat([to_elo, df], axis=0)
ratings = compute_elo_ratings(
    to_elo,
    baseline_models=baseline_models,
)
ratings = ratings[~ratings[f'model'].isin(baseline_models)]
ratings.to_csv(f'{answs_dir}/ratings.csv')
