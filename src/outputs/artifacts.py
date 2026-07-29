"""Escrita dos artefatos de uma execucao: diretorios, json, csv e log."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

from src.inputs.config import SampledParameterSet, StochasticRunConfig


class WorkspaceRepository:
    """Escreve os artefatos de uma execucao de validacao."""

    def simulation_output_dir(self, config) -> Path:
        output_dir = Path(config.paths.simulation_data) / config.simulation.name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def write_json(self, destination: Path, payload: Dict[str, Any]) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_csv(self, destination: Path, dataframe: pd.DataFrame) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(destination, index=False)


class StochasticOutputService:
    """Cria a arvore de diretorios e escreve os artefatos de um run estocastico."""

    def prepare_run_directory(self, config: StochasticRunConfig) -> dict[str, Path]:
        run_dir = Path(config.output_root) / config.run_name
        paths = {
            "run_dir": run_dir,
            "individual_runs_dir": run_dir / "individual_runs",
            "aggregated_dir": run_dir / "aggregated",
            "logs_path": run_dir / "logs.txt",
            "config_path": run_dir / "config.json",
            "sampled_parameters_path": run_dir / "sampled_parameters.csv",
            "summary_path": run_dir / "summary.json",
        }
        for key, path in paths.items():
            if key.endswith("_dir") or key == "run_dir":
                path.mkdir(parents=True, exist_ok=True)
        return paths

    def write_config(self, config: StochasticRunConfig, path: Path) -> None:
        self.write_json(path, self._to_jsonable(config))

    def write_samples(self, samples: Iterable[SampledParameterSet], path: Path) -> None:
        dataframe = pd.DataFrame([asdict(sample) for sample in samples])
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(path, index=False)

    def write_summary(self, summary: dict[str, Any], path: Path) -> None:
        self.write_json(path, summary)

    def append_log(self, path: Path, message: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{message}\n")

    def write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._to_jsonable(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _to_jsonable(self, value: Any) -> Any:
        if is_dataclass(value):
            return self._to_jsonable(asdict(value))
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): self._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(item) for item in value]
        return value

