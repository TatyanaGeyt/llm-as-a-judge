import pandas as pd
from src.calculate_ratings import compute_elo_ratings, winner
from src.utils import write_json
from pathlib import Path
import os
import argparse
from src.utils import setup_logger
import json


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
    
    baseline_models = [
        'gpt-4-1106-preview',
        'gpt-4o-mini',
        'Qwen2.5-32B-Instruct',
        'gigachat_max_26.20_uncen'
    ]
    models = [
        'llama3-70b', 
        'yagpt5lite', 
        'RuadaptQwen2.5-32B-Pro-Beta', 
        'tpro', 
        'RuadaptQwen2.5-7B-Lite-Beta', 
        'tlite', 
        'vikhr12b'
    ]

    answs_dir = Path(f"{REPO_PATH}/artifacts/judge_evaluation_results/{args.prompt}")
    model_data = {}
    for model_name in models:
        pattern = f"{model_name}_*.json"
        files = list(answs_dir.glob(pattern))
        assert files
        all_evaluations = []
        for file_path in sorted(files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_evaluations.extend(data)
            except json.JSONDecodeError as e:
                print(f"Ошибка при чтении JSON из файла {file_path}: {e}")
                continue
            except Exception as e:
                print(f"Ошибка при чтении файла {file_path}: {e}")
                continue
        model_data[model_name] = all_evaluations

    to_elo = pd.DataFrame(columns=['model', 'model_ref', 'winner'])
    for m in model_data:
        cur_res = model_data[m]
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

    model_ratings_dict = dict(zip(ratings['model'], ratings['bradley_terry_scores']))
    with open(f"{answs_dir}/ratings.json", 'w', encoding='utf-8') as f:
        json.dump(model_ratings_dict, f, indent=2, ensure_ascii=False)
