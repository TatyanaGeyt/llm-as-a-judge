"""
CoolPrompt-based autoprompting for LLM-as-a-Judge (classification A/B), aligned with
``llm_judge_autoprompt_train_val_test.ipynb``.

Requires optional dependencies: ``coolprompt``, ``langchain_openai``, ``scikit-learn``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from benchmark.judge import LLMAsJudge
from benchmark.model_instance import ModelInstance

DEFAULT_WINNER_TO_LABEL: Dict[str, str] = {"model": "A", "reference": "B"}

DEFAULT_JUDGE_LLM_CONFIG: Dict[str, Any] = {
    "temperature": 0.0,
    "max_tokens": 256,
    "top_p": 0.9,
    "n": 1,
    "stop": None,
    "timeout": 60.0,
    "max_retries": 2,
    "top_k": 40,
    "repetition_penalty": 1.0,
    "stop_token_ids": None,
    "add_generation_prompt": True,
    "skip_special_tokens": True,
    "continue_final_message": False,
    "include_stop_str_in_output": False,
    "chat_template_kwargs": {"enable_thinking": False},
}


def format_pairwise_input(
    d: dict,
    *,
    instruction_key: str = "instruction",
    model_answer_key: str = "model_answer",
    reference_answer_key: str = "reference_answer",
) -> str:
    """One labeled row → single INPUT string for CoolPrompt (A = model, B = reference)."""
    return (
        f"Запрос пользователя:\n{d[instruction_key]}\n\n"
        f"Ответ A:\n{d[model_answer_key]}\n\n"
        f"Ответ B:\n{d[reference_answer_key]}"
    )


def build_judge_llm(
    model: str,
    api_key: str,
    base_url: str,
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Build ``langchain_openai.ChatOpenAI`` from a config dict (same shape as ``JUDGE_CONFIG`` in the notebook).
    ``base_url`` should be the OpenAI-compatible root, e.g. ``http://host:6266/v1``.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError("Install langchain-openai to use autoprompt: pip install langchain-openai") from e

    c = dict(DEFAULT_JUDGE_LLM_CONFIG)
    if config:
        c.update(config)

    kwargs: Dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "base_url": base_url.rstrip("/"),
    }
    for key in ("temperature", "max_tokens", "top_p", "n", "timeout", "max_retries"):
        if c.get(key) is not None:
            kwargs[key] = c[key]
    if c.get("stop") is not None:
        kwargs["stop"] = c["stop"]

    extra_keys = (
        "top_k",
        "repetition_penalty",
        "stop_token_ids",
        "add_generation_prompt",
        "skip_special_tokens",
        "continue_final_message",
        "include_stop_str_in_output",
        "chat_template_kwargs",
    )
    extra_body = {k: c[k] for k in extra_keys if c.get(k) is not None}
    if extra_body:
        kwargs["extra_body"] = extra_body

    return ChatOpenAI(**kwargs)


def install_notebook_style_coolprompt_patches() -> Callable[[], None]:
    """
    Apply the same monkeypatches as in the notebook:

    * stratified train/val split on A/B labels;
    * DistillPrompt scores candidates on the **validation** split;
    * ``correct`` / LanguageRule disabled (no-op).
    """
    try:
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        raise ImportError("Install scikit-learn for autoprompt: pip install scikit-learn") from e

    import coolprompt.assistant as _cp_assistant
    from coolprompt.optimizer.distill_prompt.distiller import Distiller

    _PT = _cp_assistant.PromptTuner
    _orig_get_split = _PT._get_dataset_split

    def _get_dataset_split_stratified(self: Any, dataset: Any, target: Any, validation_size: float, train_as_test: bool) -> Any:
        if train_as_test:
            return _orig_get_split(self, dataset, target, validation_size, train_as_test)
        t = list(target)
        try:
            train_data, val_data, train_targets, val_targets = train_test_split(
                list(dataset),
                t,
                test_size=validation_size,
                stratify=t,
                random_state=42,
            )
        except ValueError:
            train_data, val_data, train_targets, val_targets = train_test_split(
                list(dataset),
                t,
                test_size=validation_size,
                random_state=42,
            )
        return (train_data, val_data, train_targets, val_targets)

    _PT._get_dataset_split = _get_dataset_split_stratified

    _orig_distill_eval = Distiller._evaluate

    def _distill_eval_on_validation(self: Any, prompt: str, split: str = "train") -> float:
        return _orig_distill_eval(self, prompt, split="validation")

    Distiller._evaluate = _distill_eval_on_validation

    _orig_correct = _cp_assistant.correct

    def _correct_skip(prompt: str, rule: Any, max_attempts: int = 3, **kwargs: Any) -> str:
        return prompt

    _cp_assistant.correct = _correct_skip

    def restore() -> None:
        _PT._get_dataset_split = _orig_get_split
        Distiller._evaluate = _orig_distill_eval
        _cp_assistant.correct = _orig_correct

    return restore


@dataclass
class AutopromptResult:
    """Outcome of :func:`run_autoprompt_tuning`."""

    final_prompt: str
    optimized_prompt: str
    initial_f1: Optional[float]
    final_f1: Optional[float]
    tuner: Any

    def apply_instruction_system(self, judge: LLMAsJudge) -> None:
        """Use the tuned text as the judge's system instruction (same role as in ``create_message``)."""
        judge.instruction_system = self.final_prompt


def run_autoprompt_tuning(
    rows: List[dict[str, Any]],
    *,
    target_model: ModelInstance,
    optimizer_model: ModelInstance,
    openai_base_url: str,
    start_prompt: str,
    problem_description: str,
    validation_size: float = 0.2,
    method: str = "distill",
    metric: str = "f1",
    task: str = "classification",
    train_as_test: bool = False,
    num_epochs: int = 2,
    feedback: bool = False,
    verbose: int = 2,
    winner_key: str = "winner",
    winner_to_label: Optional[Dict[str, str]] = None,
    format_row: Optional[Callable[[dict[str, Any]], str]] = None,
    instruction_key: str = "instruction",
    model_answer_key: str = "model_answer",
    reference_answer_key: str = "reference_answer",
    judge_llm_config: Optional[Dict[str, Any]] = None,
    install_patches: bool = True,
    revert_if_worse_on_val: bool = True,
    **tuner_run_kwargs: Any,
) -> AutopromptResult:
    """
    Run CoolPrompt ``PromptTuner`` like the notebook: ``target_model`` is the judge (classification),
    ``optimizer_model`` is the system model that proposes prompt edits.

    ``rows`` must include ``winner`` (``model`` / ``reference`` by default) and fields used by
    :func:`format_pairwise_input` (or your ``format_row``).
    """
    try:
        from coolprompt.assistant import PromptTuner
    except ImportError as e:
        raise ImportError("Install coolprompt to use autoprompt: pip install coolprompt") from e

    if target_model.api_base is None or optimizer_model.api_base is None:
        raise ValueError("target_model and optimizer_model must have api_base set (bind_api / ModelList).")

    api_key = target_model.api_base.key
    if optimizer_model.api_base.key != api_key:
        raise ValueError("target_model and optimizer_model must use the same API key on ApiBase.")

    wmap = dict(winner_to_label or DEFAULT_WINNER_TO_LABEL)
    fmt = format_row
    if fmt is None:

        def _default_fmt(d: dict[str, Any]) -> str:
            return format_pairwise_input(
                d,
                instruction_key=instruction_key,
                model_answer_key=model_answer_key,
                reference_answer_key=reference_answer_key,
            )

        fmt = _default_fmt

    dataset_inputs = [fmt(dict(r)) for r in rows]
    target_labels: List[str] = []
    for r in rows:
        w = r.get(winner_key)
        if w not in wmap:
            raise KeyError(f"Row id={r.get('id')}: {winner_key}={w!r} not in winner map {list(wmap.keys())}")
        target_labels.append(wmap[w])

    llm_judge = build_judge_llm(
        target_model.api_model_id,
        api_key,
        openai_base_url,
        judge_llm_config,
    )
    llm_optimizer = build_judge_llm(
        optimizer_model.api_model_id,
        api_key,
        openai_base_url,
        judge_llm_config,
    )

    restore = None
    if install_patches:
        restore = install_notebook_style_coolprompt_patches()

    tuner: Any = None
    try:
        tuner = PromptTuner(target_model=llm_judge, system_model=llm_optimizer)
        optimized_prompt = tuner.run(
            start_prompt=start_prompt,
            task=task,
            dataset=dataset_inputs,
            target=target_labels,
            method=method,
            metric=metric,
            problem_description=problem_description,
            train_as_test=train_as_test,
            validation_size=validation_size,
            num_epochs=num_epochs,
            feedback=feedback,
            verbose=verbose,
            **tuner_run_kwargs,
        )

        final_prompt = optimized_prompt
        if revert_if_worse_on_val and tuner.final_metric is not None and tuner.init_metric is not None:
            if float(tuner.final_metric) < float(tuner.init_metric):
                final_prompt = start_prompt
                tuner.final_metric = tuner.init_metric
                tuner.final_prompt = final_prompt

        return AutopromptResult(
            final_prompt=final_prompt,
            optimized_prompt=optimized_prompt,
            initial_f1=float(tuner.init_metric) if tuner.init_metric is not None else None,
            final_f1=float(tuner.final_metric) if tuner.final_metric is not None else None,
            tuner=tuner,
        )
    finally:
        if restore is not None:
            restore()
