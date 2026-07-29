from __future__ import annotations

import datetime as dt
import io
import os
import queue
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd


class QueueWriter(io.TextIOBase):
    def __init__(self, event_queue: queue.Queue):
        self._event_queue = event_queue
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self._event_queue.put({"type": "log", "message": line})
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._event_queue.put({"type": "log", "message": self._buffer.strip()})
            self._buffer = ""


def parse_float(value: str, field_name: str) -> float:
    normalized = (value or "").strip().replace(",", ".")
    if not normalized:
        raise ValueError(f"Informe um valor para {field_name}.")
    try:
        return float(normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Valor invalido para {field_name}: {value}") from exc


def validate_observed_zip(zip_path: Path) -> bool:
    if not zip_path.exists() or not zip_path.is_file():
        raise ValueError(f"Observed ZIP not found: {zip_path}")
    if zip_path.suffix.lower() != ".zip":
        raise ValueError("Observed file must be a .zip archive.")
    with zipfile.ZipFile(zip_path, "r") as archive:
        names = [item.filename.lower() for item in archive.infolist() if not item.is_dir()]
    required_exts = [".shp", ".shx", ".dbf"]
    missing = [ext for ext in required_exts if not any(name.endswith(ext) for name in names)]
    if missing:
        raise ValueError(
            "ZIP is missing required shapefile components: " + ", ".join(missing)
        )
    has_prj = any(name.endswith(".prj") for name in names)
    return has_prj


def extract_observed_bounds(zip_path: Path) -> tuple[float, float, float, float]:
    gdf = gpd.read_file(zip_path).to_crs(epsg=4326)
    min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
    return float(min_lon), float(max_lon), float(min_lat), float(max_lat)


def build_run_id() -> str:
    return dt.datetime.now().strftime("validation_%Y%m%d_%H%M%S_%f")


def stage_observed_zip(project_root: Path, run_id: str, source_zip: Path) -> Path:
    staged_dir = project_root / "runs" / run_id / "input"
    staged_dir.mkdir(parents=True, exist_ok=True)
    destination = staged_dir / source_zip.name
    shutil.copy2(source_zip, destination)
    return destination


def open_path(path: Path) -> None:
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
        return
    if os.name == "posix" and "darwin" in os.sys.platform:
        subprocess.Popen(["open", str(path)])
        return
    subprocess.Popen(["xdg-open", str(path)])


def list_environments(project_root: Path) -> list[str]:
    env_dir = project_root / "conf" / "environment"
    if not env_dir.exists():
        return ["2019"]
    names = sorted([path.stem for path in env_dir.glob("*.yaml")])
    return names or ["2019"]


def extract_metrics(out_dir: Path) -> dict:
    metrics = {
        "best_skillscore": None,
        "best_wdf": None,
        "best_cdf": None,
        "best_environmental_offset": None,
    }
    csv_path = out_dir / "wdf_cdf_optimization_fast.csv"
    if not csv_path.exists():
        return metrics

    dataframe = pd.read_csv(csv_path)
    if dataframe.empty or "skillscore" not in dataframe:
        return metrics

    valid = dataframe[dataframe["skillscore"].notna()]
    if valid.empty:
        return metrics

    best = valid.loc[valid["skillscore"].idxmax()]
    metrics["best_skillscore"] = float(best.get("skillscore"))
    metrics["best_wdf"] = float(best.get("wind_drift_factor"))
    metrics["best_cdf"] = float(best.get("current_drift_factor"))
    if "environmental_offset_hours" in best:
        metrics["best_environmental_offset"] = float(best.get("environmental_offset_hours"))
    return metrics


def build_artifact_list(out_dir: Path) -> list[Path]:
    if not out_dir.exists():
        return []
    files = [path for path in out_dir.iterdir() if path.is_file()]
    aggregated_dir = out_dir / "aggregated"
    if aggregated_dir.exists():
        files.extend(path for path in aggregated_dir.iterdir() if path.is_file())
    files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return files


def build_frame_list(out_dir: Path, sim_path: Optional[Path]) -> list[Path]:
    if sim_path is None:
        return []
    frames_dir = out_dir / f"{sim_path.stem}_frames"
    if not frames_dir.exists():
        return []
    return sorted(frames_dir.glob("*.png"))
