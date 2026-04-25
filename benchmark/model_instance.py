from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from benchmark.judge import ApiBase
from benchmark.tools import get_data, get_json_files

PathLike = Union[str, Path, None]


class ModelInstance:
    """
    One evaluated model: optional on-disk answers plus optional API client for generation.

    ``name`` is the logical id (e.g. Hugging Face id with a slash). For API payloads,
    :attr:`api_model_id` is ``name`` with only the segment after the last ``/``,
    so the inference server sees a plain model id.
    """

    def __init__(
        self,
        model_name: str,
        answers_file: str = "",
        role: str = "ev_model",
        *,
        api_base: Optional[ApiBase] = None,
        api_path: str = "",
    ) -> None:
        self.name = model_name
        self.role = role
        if isinstance(answers_file, str) and answers_file:
            self.path: PathLike = Path(answers_file)
        elif isinstance(answers_file, Path):
            self.path = answers_file
        else:
            self.path = answers_file or ""
        self.data = get_data(self.path)
        self.api_base: Optional[ApiBase] = api_base
        self.api_path = api_path

    @property
    def api_model_id(self) -> str:
        """Model id for API ``model`` field (basename after last ``/``)."""
        return self.name.rsplit("/", 1)[-1]

    def bind_api(self, api_base: ApiBase, api_path: str = "") -> None:
        """Attach or replace the API client used for this instance."""
        self.api_base = api_base
        self.api_path = api_path

    def clear_api(self) -> None:
        self.api_base = None
        self.api_path = ""

    def prepare_request_payload(self, payload: dict) -> dict:
        """Return a shallow copy of ``payload`` with ``model`` set to :attr:`api_model_id` if missing."""
        out = dict(payload)
        out.setdefault("model", self.api_model_id)
        return out

    def make_request(self, payload: dict, path: Optional[str] = None) -> dict:
        """
        POST via :attr:`api_base` (same contract as :meth:`ApiBase.make_request`).

        ``path`` defaults to this instance's ``api_path`` (then ApiBase's own default).
        """
        if self.api_base is None:
            raise RuntimeError(f"Model {self.name!r} has no api_base; call bind_api or pass default_api on ModelList.")
        req_path = self.api_path if path is None else path
        return self.api_base.make_request(self.prepare_request_payload(payload), req_path)

    def set_data(self, data: List[dict]) -> None:
        self.data = data

    def set_role(self, role: str) -> None:
        self.role = role

    def __repr__(self) -> str:
        return self.name


class ModelList:
    """
    Registry of :class:`ModelInstance` objects loaded from a directory and/or in-memory dict.

    Use :attr:`default_api` / :meth:`register_model_api` so each logical ``model_name`` can use
    its own :class:`ApiBase` (host, key, completion path).
    """

    def __init__(
        self,
        path: str = "",
        baselines: Optional[List[str]] = None,
        *,
        default_api: Optional[ApiBase] = None,
        default_api_path: str = "",
        per_model_api: Optional[Dict[str, Tuple[ApiBase, str]]] = None,
    ) -> None:
        self.path = path
        self.baselines_list: List[str] = list(baselines or [])
        self.ev_models_list: List[str] = []

        self.baselines: List[ModelInstance] = []
        self.ev_models: List[ModelInstance] = []

        self.default_api = default_api
        self.default_api_path = default_api_path
        self._model_api: Dict[str, Tuple[ApiBase, str]] = dict(per_model_api or {})

        self.load_data()

    def set(
        self,
        baselines: List[str],
        dataset_path: str = "",
        models: Optional[List[str]] = None,
        *,
        default_api: Optional[ApiBase] = None,
        default_api_path: Optional[str] = None,
    ) -> None:
        self.path = dataset_path
        self.baselines_list = list(baselines)
        self.ev_models_list = list(models or [])
        if default_api is not None:
            self.default_api = default_api
        if default_api_path is not None:
            self.default_api_path = default_api_path or ""
        self.load_data()

    def register_model_api(self, model_name: str, api_base: ApiBase, api_path: str = "") -> None:
        """Remember API settings for ``model_name`` and apply if that instance already exists."""
        self._model_api[model_name] = (api_base, api_path)
        if hasattr(self, "models") and model_name in self.models:
            self.models[model_name].bind_api(api_base, api_path)

    def _api_for(self, model_name: str) -> Tuple[Optional[ApiBase], str]:
        if model_name in self._model_api:
            base, p = self._model_api[model_name]
            return base, p
        return self.default_api, self.default_api_path

    def _new_instance(
        self,
        model_name: str,
        answers_file: PathLike,
        role: str,
    ) -> ModelInstance:
        api_base, api_path = self._api_for(model_name)
        return ModelInstance(
            model_name,
            str(answers_file) if isinstance(answers_file, Path) else (answers_file or ""),
            role,
            api_base=api_base,
            api_path=api_path,
        )

    def models_from_dataset(self, model_answers: dict) -> None:
        self.path = ""
        self.baselines_list = []
        self.ev_models_list = list(model_answers.keys())

        self.baselines = []
        self.ev_models = []
        models: Dict[str, ModelInstance] = {}
        for model_name, answ in model_answers.items():
            cur_model = self._new_instance(model_name, "", "ev_model")
            cur_model.set_data(answ)
            models[model_name] = cur_model
            self.ev_models.append(cur_model)
        self.models = models

    def reset_baselines(self, new_baselines: List[str]) -> None:
        models_list = list(self.models.keys())
        self.baselines = []
        for b in new_baselines:
            if b not in models_list:
                print(f"Model {b} not in models_list!")
                return
            self.baselines.append(self.models[b])
        self.baselines_list = new_baselines

    def load_data(self) -> None:
        self.baselines = []
        self.ev_models = []
        if not self.path:
            self.models = {}
            return
        files = get_json_files(self.path)
        self.models = self._set_models(files)

    def _set_models(self, files: List[Path]) -> Dict[str, ModelInstance]:
        models: Dict[str, ModelInstance] = {}
        ev_names: List[str] = []
        for file in files:
            model_name = file.stem
            if model_name in self.baselines_list:
                models[model_name] = self._new_instance(model_name, file, "baseline")
                self.baselines.append(models[model_name])
            elif not self.ev_models_list or model_name in self.ev_models_list:
                ev_names.append(model_name)
                models[model_name] = self._new_instance(model_name, file, "ev_model")
                self.ev_models.append(models[model_name])
        if not self.ev_models_list:
            self.ev_models_list = ev_names
        return models

    def __repr__(self) -> str:
        return (
            f"baselines | {', '.join(self.baselines_list)}\n"
            f"models    | {', '.join(self.ev_models_list)}"
        )
