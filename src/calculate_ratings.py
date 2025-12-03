from typing import List
import numpy as np
import pandas as pd
from evalica import bradley_terry, Winner
import ast


def res_token(d):
    return ast.literal_eval(d['output'])[0]


def res_prob(d): # the probability of model M winning
    if res_token(d) == 'M':
        return ast.literal_eval(d['output'])[1]
    return 1 - ast.literal_eval(d['output'])[1]


def winner(d):
    return int(res_token(d) == 'M')


def compute_elo_ratings( # taken from https://github.com/VikhrModels/ru_llm_arena/
    df: pd.DataFrame,
    baseline_models: List[str],
    initial: float = 1000.,
    base: float = 10.,
    scale: float = 400.
) -> pd.Series:
    
    df = df.copy()
    df['winner'] = df['winner'].map({
            1: Winner.X,
            0: Winner.Y
        })
    
    result = bradley_terry(
        df['model'],
        df['model_ref'],
        df['winner'],
        weights=pd.Series(1, index=range(df.shape[0])),
        solver="naive",
        tolerance=1e-8
    )
    scores = initial + np.log(result.scores) / np.log(base) * scale
    for b in baseline_models:
        if b in scores.index:
            scores += initial - scores[b]

    scores = scores.sort_values(ascending=False, kind="stable").reset_index()
    scores.columns = ['model', 'bradley_terry_scores']

    return scores


def get_rating(data): # ratio of model M wins among all responses
    M_count = sum(1 for d in data if 'M' in d['output'])
    R_count = sum(1 for d in data if 'R' in d['output'])
    assert M_count + R_count == len(data)
    return M_count / len(data)
