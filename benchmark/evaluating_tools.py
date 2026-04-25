from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef
)


def evaluate_judge(human_winners, model_winners):
    """
    human_winners: list[str]  # ground truth ('model' / 'reference')
    model_winners: list[str]  # predictions
    
    return: dict с метриками
    """

    assert len(human_winners) == len(model_winners), "Lengths must match"

    labels = ["model", "reference"]

    # --- базовые метрики ---
    acc = accuracy_score(human_winners, model_winners)

    precision, recall, f1, _ = precision_recall_fscore_support(
        human_winners,
        model_winners,
        labels=labels,
        average=None
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        human_winners,
        model_winners,
        average="macro"
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        human_winners,
        model_winners,
        average="weighted"
    )

    # --- agreement метрики ---
    kappa = cohen_kappa_score(human_winners, model_winners)
    mcc = matthews_corrcoef(human_winners, model_winners)

    # --- матрица ошибок ---
    cm = confusion_matrix(human_winners, model_winners, labels=labels)

    # --- распределения ---
    human_dist = Counter(human_winners)
    model_dist = Counter(model_winners)

    return {
        "accuracy": acc,

        "precision_per_class": dict(zip(labels, precision)),
        "recall_per_class": dict(zip(labels, recall)),
        "f1_per_class": dict(zip(labels, f1)),

        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,

        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,

        "cohen_kappa": kappa,
        "mcc": mcc,

        "confusion_matrix": cm.tolist(),

        "human_distribution": dict(human_dist),
        "model_distribution": dict(model_dist),

        "classification_report": classification_report(
            human_winners,
            model_winners,
            labels=labels
        )
    }

def prepare_eval_arrays(human_winners: dict, judge_winners: dict, verbose=True):
    """
    Возвращает:
        human_list, judge_list — списки для evaluate_judge()
    """
    common_ids = sorted(set(human_winners.keys()) & set(judge_winners.keys()))

    missing_in_judge = set(human_winners.keys()) - set(judge_winners.keys())
    extra_in_judge = set(judge_winners.keys()) - set(human_winners.keys())

    if verbose:
        print(f"Total human: {len(human_winners)}")
        print(f"Total judge: {len(judge_winners)}")
        print(f"Common: {len(common_ids)}")

        if missing_in_judge:
            print(f"⚠️ Missing in judge: {len(missing_in_judge)} (ignored)")
        if extra_in_judge:
            print(f"⚠️ Extra in judge: {len(extra_in_judge)} (ignored)")

    # --- формируем массивы ---
    human_list = [human_winners[i] for i in common_ids]
    judge_list = [judge_winners[i] for i in common_ids]

    return human_list, judge_list