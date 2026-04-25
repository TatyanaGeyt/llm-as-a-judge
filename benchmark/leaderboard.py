"""Leaderboard-related types; dataset loading and orchestration for judge pipelines."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from benchmark.judge import LLMAsJudge
from benchmark.model_instance import ModelInstance, ModelList
from benchmark.tools import make_battle

PathLike = Union[str, Path]


class LeaderboardDataset:
    """In-memory table of samples: each row is a ``dict`` (e.g. for ``LLMAsJudge.evaluate_dataset``)."""

    def __init__(self, records: Optional[List[dict[str, Any]]] = None) -> None:
        self.records: List[dict[str, Any]] = list(records) if records is not None else []

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.records)

    def to_list(self) -> List[dict[str, Any]]:
        """Return a shallow copy of all rows."""
        return list(self.records)

    @classmethod
    def from_file(
        cls,
        path: PathLike,
        *,
        file_format: Optional[str] = None,
        encoding: str = "utf-8",
    ) -> LeaderboardDataset:
        """
        Load from disk.

        * **JSON** — root must be either a list of objects, or a dict with exactly one
          key whose value is that list (e.g. ``{"train": [...]}``).
        * **JSONL / NDJSON** — one JSON object per non-empty line.

        ``file_format`` may be ``\"json\"`` or ``\"jsonl\"``; if omitted, inferred from the suffix.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(str(path))

        suffix = path.suffix.lower().lstrip(".")
        fmt = (file_format or suffix).lower()
        if fmt == "json":
            with path.open(encoding=encoding) as f:
                raw = json.load(f)
            if isinstance(raw, list):
                records = raw
            elif isinstance(raw, dict):
                if len(raw) != 1:
                    raise ValueError(
                        "JSON object root must have exactly one key mapping to a list of rows."
                    )
                sole = next(iter(raw.values()))
                if not isinstance(sole, list):
                    raise ValueError("The single value under the JSON root must be a list of objects.")
                records = sole
            else:
                raise ValueError("JSON root must be a list or a dict with one list value.")
            if not all(isinstance(row, dict) for row in records):
                raise ValueError("Every dataset row must be a JSON object (Python dict).")
            return cls(records)

        if fmt in ("jsonl", "ndjson"):
            records: List[dict[str, Any]] = []
            with path.open(encoding=encoding) as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {line_no} of {path}") from e
                    if not isinstance(row, dict):
                        raise ValueError(f"Line {line_no} must be a JSON object, got {type(row).__name__}.")
                    records.append(row)
            return cls(records)

        raise ValueError(
            f"Unknown format {fmt!r}; use 'json', 'jsonl', a path like '.jsonl', or pass file_format= explicitly."
        )

    @classmethod
    def from_huggingface(
        cls,
        path: str,
        *,
        name: Optional[str] = None,
        split: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> LeaderboardDataset:
        """
        Load via ``datasets.load_dataset`` and convert rows to plain dicts.

        If ``split`` is omitted and the hub returns a ``DatasetDict``, the split
        ``\"train\"`` is used when present, otherwise the first split in the dict.
        """
        from datasets import DatasetDict, load_dataset

        load_kwargs = dict(kwargs)
        if revision is not None:
            load_kwargs["revision"] = revision
        if token is not None:
            load_kwargs["token"] = token
        if trust_remote_code:
            load_kwargs["trust_remote_code"] = True

        obj = load_dataset(path, name=name, split=split, **load_kwargs)
        if isinstance(obj, DatasetDict):
            key = "train" if "train" in obj else next(iter(obj))
            obj = obj[key]
        if hasattr(obj, "to_list"):
            records = obj.to_list()
        else:
            records = [dict(row) for row in obj]
        return cls(records)


class Leaderboard:
    """
    Orchestrates a :class:`LeaderboardDataset`, :class:`ModelList`, and :class:`LLMAsJudge`.

    Typical flow: attach or load a dataset → :meth:`generate_all_model_answers` (optional,
    uses each model's ``ApiBase``) → :meth:`run_judge_vs_baselines` (pairs each ev model
    against baseline rows via :func:`benchmark.tools.make_battle`).
    """

    def __init__(
        self,
        judge: LLMAsJudge,
        models: ModelList,
        dataset: Optional[LeaderboardDataset] = None,
        *,
        artifacts_root: PathLike = "artifacts",
        dataset_slug: str = "dataset",
    ) -> None:
        self.judge = judge
        self.models = models
        self.dataset = dataset if dataset is not None else LeaderboardDataset()
        self.artifacts_root = Path(artifacts_root)
        self.dataset_slug = self._sanitize_slug(dataset_slug)

    @property
    def artifact_dir(self) -> Path:
        """``artifacts_root / dataset_slug`` — model answers and optional judge dumps."""
        return self.artifacts_root / self.dataset_slug

    def set_dataset(self, dataset: LeaderboardDataset, slug: Optional[str] = None) -> None:
        """Replace in-memory dataset; optionally update the slug used under ``artifacts_root``."""
        self.dataset = dataset
        if slug is not None:
            self.dataset_slug = self._sanitize_slug(slug)

    def load_dataset_from_file(
        self,
        path: PathLike,
        *,
        slug: Optional[str] = None,
        **kwargs: Any,
    ) -> LeaderboardDataset:
        """Build :class:`LeaderboardDataset` from disk and attach it (slug defaults to file stem)."""
        path = Path(path)
        ds = LeaderboardDataset.from_file(path, **kwargs)
        self.set_dataset(ds, slug=slug if slug is not None else path.stem)
        return ds

    def load_dataset_from_huggingface(
        self,
        path: str,
        *,
        slug: Optional[str] = None,
        **kwargs: Any,
    ) -> LeaderboardDataset:
        """Load via Hugging Face ``datasets``; slug defaults to a filesystem-safe form of ``path``."""
        ds = LeaderboardDataset.from_huggingface(path, **kwargs)
        default_slug = re.sub(r"[^0-9A-Za-z._-]+", "_", path.replace("/", "__"))[:200]
        self.set_dataset(ds, slug=slug if slug is not None else default_slug or "hf_dataset")
        return ds

    @staticmethod
    def _sanitize_slug(name: str) -> str:
        s = re.sub(r"[^0-9A-Za-z._-]+", "_", name.strip()).strip("._-")
        return (s[:200] if s else "dataset")

    @staticmethod
    def _safe_model_filename(name: str) -> str:
        out = re.sub(r'[\\/:*?"<>|\s]', "__", name).strip("._ ")
        return out[:180] if out else "model"

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _ensure_row_ids(rows: List[dict[str, Any]]) -> List[dict[str, Any]]:
        out: List[dict[str, Any]] = []
        for i, row in enumerate(rows):
            r = dict(row)
            r.setdefault("id", i)
            out.append(r)
        return out

    def _default_generation_payload(
        self,
        row: dict[str, Any],
        inst: ModelInstance,
        *,
        instruction_key: str,
        system_prompt: Optional[str],
        generation_extra: dict[str, Any],
    ) -> dict[str, Any]:
        text = row.get(instruction_key, "")
        messages: List[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": text if isinstance(text, str) else str(text)})
        payload: dict[str, Any] = {**generation_extra, "messages": messages}
        return payload

    def generate_all_model_answers(
        self,
        *,
        models: Optional[List[str]] = None,
        instruction_key: str = "instruction",
        system_prompt: Optional[str] = None,
        generation_extra: Optional[dict[str, Any]] = None,
        build_payload: Optional[Callable[[dict[str, Any], ModelInstance], dict[str, Any]]] = None,
    ) -> Dict[str, List[dict[str, Any]]]:
        """
        For each evaluation model with ``api_base``, call the API on every dataset row,
        write ``artifact_dir / answers_<model>.json``, and refresh ``ModelInstance.data``.

        Rows are shallow-copied and gain ``model``, ``model_answer`` (and keep ``id``).
        Models without ``api_base`` are skipped with a printed notice.
        """
        base_extra = dict(generation_extra or {})
        rows = self._ensure_row_ids(self.dataset.to_list())
        names = list(models) if models is not None else list(self.models.ev_models_list)
        out: Dict[str, List[dict[str, Any]]] = {}

        for model_name in names:
            if model_name not in self.models.models:
                raise KeyError(f"Unknown model {model_name!r} in ModelList.models")
            inst = self.models.models[model_name]
            if inst.role == "baseline":
                print(f"[Leaderboard] skip generation for {model_name!r} (baseline)")
                continue
            if inst.api_base is None:
                print(f"[Leaderboard] skip generation for {model_name!r} (no api_base)")
                continue

            built: List[dict[str, Any]] = []
            for row in rows:
                if build_payload is not None:
                    payload = build_payload(row, inst)
                else:
                    payload = self._default_generation_payload(
                        row,
                        inst,
                        instruction_key=instruction_key,
                        system_prompt=system_prompt,
                        generation_extra=base_extra,
                    )
                resp = inst.make_request(payload)
                if "error" in resp:
                    raise RuntimeError(
                        f"Generation failed for model={model_name!r}, id={row.get('id')}: {resp!r}"
                    )
                sample = dict(row)
                sample["model"] = inst.name
                sample["model_answer"] = resp.get("output", "")
                built.append(sample)

            out_path = self.artifact_dir / f"answers__{self._safe_model_filename(inst.name)}.json"
            self._write_json(out_path, built)
            inst.set_data(built)
            inst.path = str(out_path)
            out[model_name] = built

        return out

    def run_judge_with_baselines(
        self,
        *,
        ev_models: Optional[List[str]] = None,
        baseline_names: Optional[List[str]] = None,
        swapping_pos: bool = True,
        persist_judge: bool = False,
    ) -> Dict[str, Dict[str, List[Any]]]:
        """
        For each (ev, baseline) pair, merge rows by ``id`` with :func:`make_battle` and run
        :meth:`LLMAsJudge.evaluate_dataset`.

        Returns ``{ev_model_name: {baseline_name: judge_outputs}}``.
        If ``persist_judge``, writes JSON under ``artifact_dir / judge /``.
        """
        if not self.judge.prompt.set:
            raise RuntimeError("Judge prompt is not set; call judge.set_prompt(...) first.")

        ev_names = list(ev_models) if ev_models is not None else list(self.models.ev_models_list)
        bl_names = list(baseline_names) if baseline_names is not None else list(self.models.baselines_list)
        if not ev_names:
            raise ValueError("No evaluation models; set ModelList.ev_models_list or pass ev_models=.")
        if not bl_names:
            raise ValueError("No baselines in ModelList; set baselines when building ModelList.")

        results: Dict[str, Dict[str, List[Any]]] = {}
        judge_dir = self.artifact_dir / "judge"

        for ev in ev_names:
            if ev not in self.models.models:
                raise KeyError(f"Unknown ev model {ev!r}")
            ev_inst = self.models.models[ev]
            results[ev] = {}
            for bl in bl_names:
                if bl not in self.models.models:
                    raise KeyError(f"Unknown baseline model {bl!r}")
                bl_inst = self.models.models[bl]
                battle = make_battle(ev_inst.data, bl_inst.data)
                if not battle:
                    raise ValueError(
                        f"No aligned id overlap between ev={ev!r} and baseline={bl!r}; "
                        "ensure both ModelInstance.data lists share the same id values."
                    )
                raw = self.judge.evaluate_dataset(battle, swapping_pos=swapping_pos)
                results[ev][bl] = raw
                if persist_judge:
                    fn = f"{self._safe_model_filename(ev)}__vs__{self._safe_model_filename(bl)}.json"
                    self._write_json(judge_dir / fn, raw)

        return results

    def tune_judge_autoprompt(
        self,
        target_model: ModelInstance,
        optimizer_model: ModelInstance,
        *,
        openai_base_url: str,
        start_prompt: str,
        problem_description: str,
        persist_result: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Run CoolPrompt tuning on :attr:`dataset` (same protocol as the autoprompt notebook).

        ``target_model`` / ``optimizer_model`` are :class:`~benchmark.model_instance.ModelInstance`
        objects with ``api_base`` (and typically ``api_path``) so this code can build LangChain
        clients with :attr:`~benchmark.model_instance.ModelInstance.api_model_id`.

        ``openai_base_url`` must be the OpenAI-compatible API root, e.g. ``http://host:6266/v1``.

        Returns :class:`~benchmark.autoprompt.AutopromptResult`. If ``persist_result``, writes
        ``artifact_dir / autoprompt_result.json`` (prompt + F1 fields, without the live ``tuner``).
        """
        from benchmark.autoprompt import AutopromptResult, run_autoprompt_tuning

        res: AutopromptResult = run_autoprompt_tuning(
            self.dataset.to_list(),
            target_model=target_model,
            optimizer_model=optimizer_model,
            openai_base_url=openai_base_url,
            start_prompt=start_prompt,
            problem_description=problem_description,
            **kwargs,
        )
        if persist_result:
            self._write_json(
                self.artifact_dir / "autoprompt_result.json",
                {
                    "final_prompt": res.final_prompt,
                    "optimized_prompt": res.optimized_prompt,
                    "initial_f1": res.initial_f1,
                    "final_f1": res.final_f1,
                },
            )
        return res
