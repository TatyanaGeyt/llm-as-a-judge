import pandas as pd
from src.calculate_ratings import compute_elo_ratings, winner, res_token, res_prob
from src.style_control import compute_style_control, get_element_counts, scale_to_elo
from src.utils import write_json
from pathlib import Path
import os
import argparse
from src.utils import setup_logger
import json


REPO_PATH = os.environ['REPO_PATH']

def make_style_control(to_elo, baseline, log):
    log.info(f'[style_control] calculate styles for answers')
    model_dir = Path(f"{REPO_PATH}/datasets/ru_arena-hard-v0.1/model_answers")
    pattern = f"*.json"
    files = list(model_dir.glob(pattern))
    baseline_dir = Path(f"{REPO_PATH}/datasets/ru_arena-hard-v0.1/baselines")
    pattern = f"*.json"
    files += list(baseline_dir.glob(pattern))

    style_data = {}
    for file_path in sorted(files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                model_name = file_path.stem
                style_data[model_name] = [
                        {
                            'id': i,
                            'style': get_element_counts(d['output'])
                        } for i, d in enumerate(data)
                    ]
        except Exception as e:
            log.error(f'{e}')
    
    log.info(f'[style_control] appending styles in dataframe')
    model_a_style = []
    model_b_style = []
    for _, row in to_elo.iterrows():
        id = row['id']
        model_a = row['model_a']
        cur_model_a_style = [s['style'] for s in style_data[model_a] if s['id'] == id][0]
        model_b = row['model_b']
        cur_model_b_style = [s['style'] for s in style_data[model_b] if s['id'] == id][0]
        model_a_style.append(cur_model_a_style)
        model_b_style.append(cur_model_b_style)

    to_elo['model_a_style'] = model_a_style
    to_elo['model_b_style'] = model_b_style

    log.info(f'[style_control] start computing style control')
    ratings, models = compute_style_control(to_elo)
    log.info(f'[style_control] scaling ratings, baseline = {baseline}')
    ratings = scale_to_elo(ratings, models, baseline_model=baseline)

    return ratings, models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Setting environment variables for a project')
    
    parser.add_argument(
        '--prompt',
        required=True, 
        type=str,
        help='prompt given to the Judge'
    )
    parser.add_argument(
        '--style',
        required=False, 
        action='store_true',
        default=False,
        help='Enable style control mode'
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
                log.info(f"Ошибка при чтении JSON из файла {file_path}: {e}")
                continue
            except Exception as e:
                log.info(f"Ошибка при чтении файла {file_path}: {e}")
                continue
        model_data[model_name] = all_evaluations

    to_elo = pd.DataFrame(columns=['model', 'model_ref', 'winner'])
    log.info(f'Start swapping possition analysis')

    for m in model_data:
        cur_res = model_data[m]
        assert len(cur_res) % 2 == 0
        id = []
        model_ref = []
        model = []
        win = []
        for i in range(0, len(cur_res), 2):
            sample = cur_res[i]
            sample_rev = cur_res[i+1]
            try:
                assert sample['m'] == sample_rev['r'] and sample['r'] == sample_rev['m']
            except Exception as e:
                log.error(f"[{sample['id']}, {sample['m']} & {sample['r']}] error not reverse: {sample}, {sample_rev}")
            
            id.append(sample['id'])
            model.append(sample['m'])
            model_ref.append(sample['r'])
            if res_token(sample) != res_token(sample_rev):
                win.append(winner(sample))
            elif res_prob(sample) > res_prob(sample_rev):
                win.append(winner(sample))
            else:
                win.append(winner(sample_rev))

        df = pd.DataFrame()
        df['id'] = id
        df['model_ref'] = model_ref
        df['model'] = model
        df['winner'] = win
        to_elo = pd.concat([to_elo, df], axis=0)
    
    if args.style:
        to_elo = to_elo.rename(columns={'model': 'model_a', 'model_ref': 'model_b'})
        ratings, models = make_style_control(to_elo, baseline='gpt-4-1106-preview', log=log)

    else:
        ratings = compute_elo_ratings(
            to_elo,
            baseline_models=baseline_models,
        )
        ratings = ratings[~ratings[f'model'].isin(baseline_models)]
        ratings.to_csv(f'{answs_dir}/ratings.csv')

        models = ratings['model']
        ratings = ratings['bradley_terry_scores']

    model_ratings_dict = dict(zip(models, ratings))
    with open(f"{answs_dir}/ratings.json", 'w', encoding='utf-8') as f:
        json.dump(model_ratings_dict, f, indent=2, ensure_ascii=False)
