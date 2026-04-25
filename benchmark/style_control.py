from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from scipy.special import expit
from tqdm import tqdm
from scipy.optimize import minimize
from math import log
import re


# taken from https://github.com/lm-sys/FastChat

tqdm.pandas()

STYLE_CONTROL_ELEMENTS = [
    "len_answer",
    "header_count",
    "list_count",
    "bold_count",
    "code_blocks_count"
]

# Length-only contextual feature (same ``len_answer`` definition as in full style counts).
LENGTH_CONTROL_ELEMENTS: List[str] = ["len_answer"]

# Per-feature std from train can be ~0 for nearly-constant style ratios; z-scoring then
# explodes on held-out pairs and saturates sigmoid. Floor keeps train/apply consistent.
STYLE_DIFF_STD_FLOOR = 0.05
# Cap z-scores so OOD style ratios do not dominate the logit (train |z| is typically < 4).
STYLE_FEATURE_CLIP = 4.0


def _clip_style_z(z: np.ndarray) -> np.ndarray:
    return np.clip(z, -STYLE_FEATURE_CLIP, STYLE_FEATURE_CLIP)

DIFF_MASK = np.array([1.0, -1.0], dtype=np.float64)

def count_style_elements(markdown_text):
    def remove_pattern(answer, pattern):
        blocks = pattern.findall(answer)
        for block in blocks:
            answer = answer.replace(block, "")
        return answer

    len_answer = len(markdown_text)
    code_count = len(re.findall(r"```[^`]+```", markdown_text))
    code_pattern = re.compile("```([^`]+)```")
    markdown_text = remove_pattern(markdown_text, code_pattern)
    markdown_text = markdown_text.replace("```", "")

    mono_count = len(re.findall(r"`[^`]+`", markdown_text))
    mono_pattern = re.compile("`([^`]+)`")
    markdown_text = remove_pattern(markdown_text, mono_pattern)
    counters = {
        f"len_answer": len_answer,
        f"header_count": {
            "h1": len(re.findall(r"^#{1}\s", markdown_text, re.MULTILINE)),
            "h2": len(re.findall(r"^#{2}\s", markdown_text, re.MULTILINE)),
            "h3": len(re.findall(r"^#{3}\s", markdown_text, re.MULTILINE)),
            "h4": len(re.findall(r"^#{4}\s", markdown_text, re.MULTILINE)),
            "h5": len(re.findall(r"^#{5}\s", markdown_text, re.MULTILINE)),
            "h6": len(re.findall(r"^#{6}\s", markdown_text, re.MULTILINE)),
        },
        f"list_count": {
            "ordered": len(re.findall(r"^\s*\d+\.\s", markdown_text, re.MULTILINE)),
            "unordered": len(re.findall(r"^\s*[-*+]\s", markdown_text, re.MULTILINE)),
        },
        f"bold_count": {
            "**": len(re.findall(r"\*\*[^*\n]+\*\*", markdown_text)),
            "__": len(re.findall(r"__[^_\n]+__", markdown_text)),
        },
        f"code_blocks_count": {
            "`": mono_count,
            "```": code_count,
        },
    }
    return counters


def extract_style_feature(x, feature):
    val = x[feature]
    if isinstance(val, int):
        return val
    else:
        return sum(val.values())


def get_element_counts(text):
    style_elements = count_style_elements(text)
    el_counts = []
    for feature in style_elements:
        el_counts.append(extract_style_feature(style_elements, feature))
    return el_counts


def get_length_control_counts(text: str) -> List[float]:
    """
    Single-feature counts for length-only BT (raw ``len_answer``, identical to the first
    component of :func:`get_element_counts`).
    """
    counters = count_style_elements(text)
    return [float(extract_style_feature(counters, "len_answer"))]


def _style_diff_matrix_raw(
    model_a: pd.Series,
    model_b: pd.Series,
    style_elements: List[str] = STYLE_CONTROL_ELEMENTS,
) -> np.ndarray:
    """Relative style difference per feature, shape (n_features, n_battles)."""
    n_features = len(style_elements)
    n_battles = model_a.shape[0]
    style_matrix = np.zeros(shape=(2 * n_features, n_battles), dtype=np.float64)
    for idx, element in enumerate(style_elements):
        style_matrix[idx, :] = np.array([el[idx] for el in model_a], dtype=np.float64)
    for idx, element in enumerate(style_elements):
        style_matrix[n_features + idx, :] = np.array([el[idx] for el in model_b], dtype=np.float64)
    style_diff = (style_matrix[:n_features] - style_matrix[n_features]).astype(np.float64)
    style_sum = (style_matrix[:n_features] + style_matrix[n_features]).astype(np.float64)
    style_diff /= np.maximum(style_sum, 1e-12)
    return style_diff


def calculate_style_with_stats(
    model_a: pd.Series,
    model_b: pd.Series,
    style_elements: List[str] = STYLE_CONTROL_ELEMENTS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalized style features (n_battles, n_features) plus train-time mean/std per feature
    (for applying the same normalization to a new pair).
    """
    style_diff = _style_diff_matrix_raw(model_a, model_b, style_elements)
    style_mean = np.mean(style_diff, axis=1)
    style_std = np.std(style_diff, axis=1)
    style_std = np.maximum(style_std, STYLE_DIFF_STD_FLOOR)
    z = (style_diff - style_mean[:, np.newaxis]) / style_std[:, np.newaxis]
    features = _clip_style_z(z).T
    return features, style_mean, style_std


def calculate_style(
    model_a: pd.Series,
    model_b: pd.Series,
    style_elements: List[str] = STYLE_CONTROL_ELEMENTS,
) -> np.ndarray:
    features, _, _ = calculate_style_with_stats(model_a, model_b, style_elements)
    return features


def normalized_style_features_for_counts(
    counts_a: List[float],
    counts_b: List[float],
    style_mean: np.ndarray,
    style_std: np.ndarray,
) -> np.ndarray:
    """Single-pair normalized feature vector (same pipeline as ``calculate_style_with_stats``)."""
    v_a = np.asarray(counts_a, dtype=np.float64)
    v_b = np.asarray(counts_b, dtype=np.float64)
    style_diff = (v_a - v_b) / np.maximum(v_a + v_b, 1e-12)
    std = np.maximum(np.asarray(style_std, dtype=np.float64), STYLE_DIFF_STD_FLOOR)
    z = (style_diff - np.asarray(style_mean, dtype=np.float64)) / std
    return _clip_style_z(z)


def winner_labels_to_outcomes(
    winners: List[Union[int, float, str, bool]],
    n_rows: int,
) -> np.ndarray:
    """Map parallel ``winners`` to 1.0 = model A won, 0.0 = model B (reference) won."""
    if len(winners) != n_rows:
        raise ValueError(f"winners length {len(winners)} != dataset length {n_rows}")
    out = np.zeros(n_rows, dtype=np.float64)
    for i, w in enumerate(winners):
        if isinstance(w, (bool, np.bool_)):
            out[i] = 1.0 if w else 0.0
        elif isinstance(w, (int, float, np.integer, np.floating)):
            fv = float(w)
            if fv not in (0.0, 1.0):
                raise ValueError(f"Numeric winner must be 0 or 1, got {w!r} at index {i}")
            out[i] = fv
        else:
            s = str(w).strip().lower()
            if s in ("model", "a", "left", "first", "1", "true", "yes"):
                out[i] = 1.0
            elif s in ("reference", "b", "right", "second", "0", "false", "no"):
                out[i] = 0.0
            elif s in ("tie", "draw", "equal"):
                raise ValueError(
                    f"Tie outcomes are not supported for contextual BT training (index {i}); drop or remap rows."
                )
            else:
                raise ValueError(f"Unsupported winner label {w!r} at index {i}")
    return out


def judge_label_to_side_sign(label: Any) -> float:
    """
    Map a judge verdict to a logit offset direction: +1 favours model (A), -1 favours reference (B), 0 neutral.
    Accepts numeric probability of A winning (uses >0.5 / <0.5), or strings like model/reference/A/B.
    """
    if label is None:
        return 0.0
    if isinstance(label, (bool, np.bool_)):
        return 1.0 if label else -1.0
    if isinstance(label, (int, float, np.integer, np.floating)):
        v = float(label)
        if 0.0 <= v <= 1.0 and v not in (0.0, 1.0):
            if v > 0.5:
                return 1.0
            if v < 0.5:
                return -1.0
            return 0.0
        if v == 1.0:
            return 1.0
        if v == 0.0:
            return -1.0
        return 0.0
    s = str(label).strip().lower()
    if s in ("model", "a", "left", "first", "1", "true", "yes"):
        return 1.0
    if s in ("reference", "b", "right", "second", "0", "false", "no"):
        return -1.0
    if s in ("tie", "draw", "equal"):
        return 0.0
    return 0.0


def fit_style_control_state(
    df: pd.DataFrame,
    *,
    alpha: float = log(10.0),
    reg: float = 10.0,
    tol: float = 1e-6,
    style_elements: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Fit contextual BT + style (FastChat) on a dataframe with columns
    ``model_a``, ``model_b``, ``model_a_style``, ``model_b_style``, ``winner`` (0/1).

    ``style_elements`` selects which coordinates each row of ``*_style`` encodes (default:
    full :data:`STYLE_CONTROL_ELEMENTS`). Use :data:`LENGTH_CONTROL_ELEMENTS` for length-only.

    ``reg`` L2-penalizes all parameters (including style weights). Too-small ``reg`` lets
    ``w·f`` dominate on held-out pairs when style z-scores shift OOD, saturating ``sigmoid``.
    """
    elements = style_elements if style_elements is not None else STYLE_CONTROL_ELEMENTS
    features, style_mean, style_std = calculate_style_with_stats(
        df.model_a_style, df.model_b_style, style_elements=elements
    )
    matchups, models = get_matchups_models(df.model_a, df.model_b)
    outcomes = np.asarray(df.winner.values, dtype=np.float64)
    params = fit_contextual_bt(
        matchups,
        features,
        outcomes,
        models=models,
        alpha=alpha,
        reg=reg,
        tol=tol,
    )
    return {
        "models": models,
        "params": np.asarray(params, dtype=np.float64),
        "style_mean": np.asarray(style_mean, dtype=np.float64),
        "style_std": np.asarray(style_std, dtype=np.float64),
        "alpha": float(alpha),
        "reg": float(reg),
        "tol": float(tol),
        "style_elements": list(elements),
    }


def fit_length_control_state(
    df: pd.DataFrame,
    *,
    alpha: float = log(10.0),
    reg: float = 10.0,
    tol: float = 1e-6,
) -> Dict[str, Any]:
    """Same as :func:`fit_style_control_state` with a single contextual feature: answer length."""
    return fit_style_control_state(
        df, alpha=alpha, reg=reg, tol=tol, style_elements=LENGTH_CONTROL_ELEMENTS
    )


def apply_style_control_state(
    state: Dict[str, Any],
    *,
    model_a_name: str,
    model_b_name: str,
    counts_a: List[float],
    counts_b: List[float],
    judge_label: Any = None,
    judge_strength: float = 1.0,
) -> float:
    """
    FastChat-style probability that side A (model_a) wins for one pair:
    ``sigmoid(alpha*(r_a-r_b) + w·f + judge_strength * sign(judge_label))``.
    Unknown models use rating 0 (same as all-zero init in ``fit_contextual_bt``).
    """
    models: List[str] = state["models"]
    params = state["params"]
    n_m = len(models)
    ratings = params[:n_m]
    w = params[n_m:]
    f = normalized_style_features_for_counts(counts_a, counts_b, state["style_mean"], state["style_std"])

    def _rating(name: str) -> float:
        if name not in models:
            return 0.0
        return float(ratings[models.index(name)])

    ra, rb = _rating(model_a_name), _rating(model_b_name)
    logit = state["alpha"] * (ra - rb) + float(np.dot(w, f))
    if judge_label is not None and judge_strength != 0.0:
        logit += judge_strength * judge_label_to_side_sign(judge_label)
    return float(expit(logit))


def get_matchups_models(model_a: pd.Series, model_b: pd.Series):
    n_rows = len(model_a)
    assert len(model_b) == n_rows
    model_indices, models = pd.factorize(pd.concat([model_a, model_b]))
    matchups = np.column_stack([model_indices[:n_rows], model_indices[n_rows:]])
    return matchups, models.to_list()


def contextual_bt_loss_and_grad(
    params,
    n_competitors,
    matchups,
    features,
    outcomes,
    alpha=1.0,
    reg=1.0,
    half_reg=0.5,
):
    reg_loss = half_reg * np.inner(params, params)

    ratings = params[:n_competitors]
    feature_params = params[n_competitors:]

    matchup_ratings = ratings[matchups]

    bt_logits = alpha * (matchup_ratings[:, 0] - matchup_ratings[:, 1])
    context_logits = np.dot(features, feature_params)
    probs = expit(bt_logits + context_logits)
    loss = (
        -((np.log(probs) * outcomes + np.log(1.0 - probs) * (1.0 - outcomes))).sum()
        + reg_loss
    )

    error = outcomes - probs
    grad = reg * params
    matchups_grads = -alpha * error
    np.add.at(
        grad[:n_competitors], matchups[:, [0, 1]], matchups_grads[:, None] * DIFF_MASK
    )
    grad = grad.astype(np.float64)
    features = features.astype(np.float64)
    error = error.astype(np.float64)
    
    grad[n_competitors:] -= np.dot(features.T, error)

    return loss, grad


def fit_contextual_bt(
    matchups,
    features,
    outcomes,
    models,
    idxs=None,
    alpha=log(10.0),
    reg=10.0,
    tol=1e-6,
):
    n_features = features.shape[1]
    n_models = len(models)
    initial_params = np.zeros(n_models + n_features, dtype=np.float64)
    half_reg = reg / 2.0

    if idxs is not None:
        matchups, features, outcomes = matchups[idxs], features[idxs], outcomes[idxs]

    result = minimize(
        fun=contextual_bt_loss_and_grad,
        x0=initial_params,
        args=(n_models, matchups, features, outcomes, alpha, reg, half_reg),
        jac=True,
        method="L-BFGS-B",
        options={"disp": False, "maxiter": 100, "gtol": tol},
    )
    return result["x"]


def compute_style_control(
    df: pd.DataFrame,
    alpha=log(10.0), reg=10.0, tol=1e-6
):
    features = calculate_style(df.model_a_style, df.model_b_style)
    matchups, models = get_matchups_models(df.model_a, df.model_b)
    outcomes = df.winner.values
    params = fit_contextual_bt(
        matchups,
        features,
        outcomes,
        models=models,
        alpha=alpha,
        reg=reg,
        tol=tol,
    )
    ratings = params[: len(models)]
    return ratings, models

def scale_to_elo(
    ratings,
    models,
    baseline_model='',
    baseline_rating=1000, # Стандартная начальная точка для Elo
    scale_factor=400.0,
):
    elo_ratings = scale_factor * ratings
    
    if baseline_model and baseline_model in models:
        baseline_idx = models.index(baseline_model)
        # Смещаем все рейтинги так, чтобы у базовой модели был рейтинг baseline_rating
        offset = baseline_rating - elo_ratings[baseline_idx]
        elo_ratings += offset
        
    return elo_ratings