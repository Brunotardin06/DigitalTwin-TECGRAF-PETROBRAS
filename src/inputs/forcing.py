"""Dados ambientais de forcing: baixar, adaptar, validar e virar readers.

Cobre o caminho completo dos arquivos de corrente, vento e sal/temperatura
ate o ponto em que o OpenDrift consegue le-los:

  * CopernicusGateway         baixa os datasets do YAML do ambiente
  * ForcingDatasetAdapter     conserta datasets REMO para os readers CF
  * validate_environment_coverage  confere se as manchas caem na cobertura
  * build_readers             monta os readers ja com os offsets aplicados
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Callable, Optional, Sequence

import copernicusmarine as cm
import numpy as np
import pandas as pd
import xarray as xr
from opendrift.readers import reader_netCDF_CF_generic

# Os datasets do Copernicus vem em UTC e as manchas observadas em horario
# local (UTC-3); o offset alinha os dois. Pode ser sobrescrito por execucao.
DEFAULT_ENVIRONMENTAL_OFFSET_HOURS = -3.0
MAX_ENVIRONMENTAL_OFFSET_HOURS = 24.0
SUPPORTED_FORCING_SOURCES = {"COPERNICUS", "NOAA", "REMO"}


# --------------------------------------------------------------------------
# Normalizacao de entradas
# --------------------------------------------------------------------------


def normalize_forcing_source(value: Optional[str]) -> str:
    source = (value or "COPERNICUS").strip().upper()
    if source not in SUPPORTED_FORCING_SOURCES:
        raise ValueError(
            f"Unsupported forcing source '{value}'. Supported values: COPERNICUS, NOAA, REMO."
        )
    return source


def normalize_path_list(
    singular_path: Optional[str],
    multiple_paths: Optional[list[str]] = None,
) -> list[Path]:
    values: list[str] = []
    if multiple_paths:
        values.extend(str(path).strip() for path in multiple_paths if str(path).strip())
    if singular_path and str(singular_path).strip():
        values.append(str(singular_path).strip())
    return list(dict.fromkeys(Path(value) for value in values))


def normalize_environmental_offset_values(
    values: Optional[list[float] | tuple[float, ...]],
) -> Optional[tuple[float, ...]]:
    if values is None:
        return None

    normalized: list[float] = []
    for value in values:
        offset = float(value)
        if not np.isfinite(offset):
            raise ValueError("environmental offset values must be finite")
        if abs(offset) > MAX_ENVIRONMENTAL_OFFSET_HOURS:
            raise ValueError(
                f"environmental offset values must be between "
                f"-{MAX_ENVIRONMENTAL_OFFSET_HOURS:g} and {MAX_ENVIRONMENTAL_OFFSET_HOURS:g}"
            )
        normalized.append(offset)

    if not normalized:
        return None
    return tuple(dict.fromkeys(normalized))


# --------------------------------------------------------------------------
# Validacao de cobertura espaco-temporal
# --------------------------------------------------------------------------


def _dataset_coord_range(ds, names: tuple[str, ...]) -> tuple[float, float]:
    for name in names:
        if name in ds.coords or name in ds.variables:
            values = np.asarray(ds[name].values, dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                return float(finite.min()), float(finite.max())
    raise ValueError(f"Could not find coordinate names {names} in dataset.")


def _has_overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> bool:
    return not (a_max < b_min or a_min > b_max)


def validate_environment_coverage(
    manchas,
    current_dataset_paths: tuple[Path, ...],
    sal_temp_dataset_path: Path,
    wind_dataset_paths: tuple[Path, ...] = (),
    environmental_offset_hours: float = 0.0,
) -> None:
    """Levanta ValueError se as manchas caem fora da cobertura dos datasets."""
    obs_min_lon, obs_min_lat, obs_max_lon, obs_max_lat = [float(v) for v in manchas.total_bounds]
    obs_times = pd.to_datetime(manchas["datetime"], errors="coerce").dropna()
    obs_min_time = obs_times.min() if not obs_times.empty else None
    obs_max_time = obs_times.max() if not obs_times.empty else None

    required_groups: list[tuple[str, tuple[Path, ...]]] = [
        ("current", tuple(current_dataset_paths)),
        ("sal_temp", (Path(sal_temp_dataset_path),)),
    ]
    if wind_dataset_paths:
        required_groups.append(("wind", tuple(wind_dataset_paths)))

    for name, dataset_paths in required_groups:
        if not dataset_paths:
            raise ValueError(f"At least one dataset path is required for '{name}'.")

        agg_min_lon = float("inf")
        agg_max_lon = float("-inf")
        agg_min_lat = float("inf")
        agg_max_lat = float("-inf")
        agg_min_time: Optional[pd.Timestamp] = None
        agg_max_time: Optional[pd.Timestamp] = None

        for dataset_path in dataset_paths:
            if not dataset_path.exists():
                raise ValueError(
                    f"Required dataset not found: {dataset_path}. "
                    "Provide current/wind files and download sal_temp before execution."
                )
            with xr.open_dataset(dataset_path) as ds:
                ds_min_lon, ds_max_lon = _dataset_coord_range(ds, ("longitude", "lon", "x"))
                ds_min_lat, ds_max_lat = _dataset_coord_range(ds, ("latitude", "lat", "y"))
                agg_min_lon = min(agg_min_lon, ds_min_lon)
                agg_max_lon = max(agg_max_lon, ds_max_lon)
                agg_min_lat = min(agg_min_lat, ds_min_lat)
                agg_max_lat = max(agg_max_lat, ds_max_lat)

                time_name = None
                for candidate in ("time", "time1"):
                    if candidate in ds.coords or candidate in ds.variables:
                        time_name = candidate
                        break
                if time_name is not None:
                    ds_times = pd.to_datetime(ds[time_name].values, errors="coerce")
                    ds_times = ds_times[~pd.isna(ds_times)]
                    if len(ds_times):
                        if environmental_offset_hours:
                            ds_times = ds_times + pd.Timedelta(hours=float(environmental_offset_hours))
                        ds_min_time = ds_times.min()
                        ds_max_time = ds_times.max()
                        agg_min_time = ds_min_time if agg_min_time is None else min(agg_min_time, ds_min_time)
                        agg_max_time = ds_max_time if agg_max_time is None else max(agg_max_time, ds_max_time)

        if not _has_overlap(obs_min_lon, obs_max_lon, agg_min_lon, agg_max_lon) or not _has_overlap(
            obs_min_lat, obs_max_lat, agg_min_lat, agg_max_lat
        ):
            raise ValueError(
                f"Observed spill area is outside '{name}' dataset coverage. "
                f"Observed lon/lat=[{obs_min_lon:.5f},{obs_max_lon:.5f}] / "
                f"[{obs_min_lat:.5f},{obs_max_lat:.5f}], "
                f"dataset lon/lat=[{agg_min_lon:.5f},{agg_max_lon:.5f}] / "
                f"[{agg_min_lat:.5f},{agg_max_lat:.5f}]."
            )

        if obs_min_time is not None and obs_max_time is not None and agg_min_time is not None and agg_max_time is not None:
            if obs_max_time < agg_min_time or obs_min_time > agg_max_time:
                raise ValueError(
                    f"Observed spill time window is outside '{name}' dataset time coverage. "
                    f"Observed=[{obs_min_time},{obs_max_time}], "
                    f"dataset=[{agg_min_time},{agg_max_time}]."
                )


def resolve_forcing_paths(
    config,
    forcing_source: str,
    adapter: "ForcingDatasetAdapter",
    current_dataset_path: Optional[str] = None,
    wind_dataset_path: Optional[str] = None,
    current_dataset_paths: Optional[Sequence[str]] = None,
    wind_dataset_paths: Optional[Sequence[str]] = None,
) -> tuple[list[Path], list[Path], Path]:
    """Decide quais arquivos de forcing usar e prepara cada um para leitura.

    A lista plural tem prioridade sobre a singular; sem nenhuma das duas, cai
    no water_dataset_path do YAML. Vento e opcional (lista vazia e valida).

    Retorna (correntes, ventos, sal_temp), todos ja passados pelo adapter.
    """
    if current_dataset_paths:
        current_paths = [Path(path) for path in current_dataset_paths]
    elif current_dataset_path:
        current_paths = [Path(current_dataset_path)]
    else:
        current_paths = [Path(config.copernicusmarine.specificities.water_dataset_path)]

    if wind_dataset_paths:
        wind_paths = [Path(path) for path in wind_dataset_paths]
    elif wind_dataset_path:
        wind_paths = [Path(wind_dataset_path)]
    else:
        wind_paths = []

    source = normalize_forcing_source(forcing_source)
    current_paths = [adapter.prepare_path(path, source, "current") for path in current_paths]
    wind_paths = [adapter.prepare_path(path, source, "wind") for path in wind_paths]
    sal_temp_path = Path(config.copernicusmarine.specificities.sal_temp_dataset_path)
    return current_paths, wind_paths, sal_temp_path


def build_reader(
    dataset_path: Path,
    environmental_offset_hours: Optional[float] = None,
    temporal_lag_seconds: Optional[float] = None,
):
    """Abre um dataset como reader do OpenDrift com o eixo de tempo ajustado.

    O offset alinha o forcing ao horario das manchas. O lag temporal (usado
    pelo ensemble estocastico) descreve um forcing amostrado em
    `tempo_simulacao + lag`; deslocar o eixo em -lag torna esse campo
    disponivel no instante de simulacao original.
    """
    reader = reader_netCDF_CF_generic.Reader(dataset_path)
    offset_hours = (
        DEFAULT_ENVIRONMENTAL_OFFSET_HOURS
        if environmental_offset_hours is None
        else float(environmental_offset_hours)
    )
    if temporal_lag_seconds is not None:
        reader.shift_start_time(
            reader.start_time
            + timedelta(hours=offset_hours)
            - timedelta(seconds=float(temporal_lag_seconds))
        )
    elif offset_hours:
        reader.shift_start_time(reader.start_time + timedelta(hours=offset_hours))
    return reader


def build_readers(
    current_paths: Sequence[Path],
    wind_paths: Sequence[Path],
    sal_temp_path: Optional[Path] = None,
    environmental_offset_hours: Optional[float] = None,
    temporal_lag_seconds: Optional[float] = None,
) -> list:
    """Monta a lista de readers na ordem de prioridade que o OpenDrift usa.

    Correntes vem antes de ventos e, dentro de cada grupo, os arquivos mais
    recentes primeiro: quando duas rodadas de forecast se sobrepoem no tempo,
    o OpenDrift consulta o primeiro reader que cobre o instante pedido.
    """
    def make(path):
        return build_reader(
            path,
            environmental_offset_hours=environmental_offset_hours,
            temporal_lag_seconds=temporal_lag_seconds,
        )

    current_readers = sorted(
        (make(path) for path in current_paths),
        key=lambda reader: reader.start_time,
        reverse=True,
    )
    wind_readers = sorted(
        (make(path) for path in wind_paths),
        key=lambda reader: reader.start_time,
        reverse=True,
    )
    readers = current_readers + wind_readers
    if sal_temp_path is not None:
        readers.append(make(sal_temp_path))
    return readers


def valid_environmental_offsets(
    manchas,
    current_dataset_paths: tuple[Path, ...],
    sal_temp_dataset_path: Path,
    wind_dataset_paths: tuple[Path, ...],
    environmental_offset_values: tuple[float, ...],
) -> tuple[float, ...]:
    """Filtra os offsets que tem cobertura valida. Levanta se nenhum sobrar."""
    valid_offsets: list[float] = []
    first_error: Optional[Exception] = None
    for offset in environmental_offset_values:
        try:
            validate_environment_coverage(
                manchas,
                current_dataset_paths=current_dataset_paths,
                wind_dataset_paths=wind_dataset_paths,
                sal_temp_dataset_path=sal_temp_dataset_path,
                environmental_offset_hours=offset,
            )
            valid_offsets.append(offset)
        except ValueError as exc:
            if first_error is None:
                first_error = exc
            print(f"Skipping environmental offset {offset:g} h: {exc}")

    if not valid_offsets:
        raise ValueError(
            "No environmental offset has valid forcing coverage."
            + (f" First error: {first_error}" if first_error else "")
        )
    return tuple(valid_offsets)


# --------------------------------------------------------------------------
# Adaptacao de datasets para os readers do OpenDrift
# --------------------------------------------------------------------------


class ForcingDatasetAdapter:
    """Adapta datasets de terceiros ao formato que os readers CF esperam.

    Hoje so o REMO precisa de tratamento (renomear time1, remover a dimensao
    depth unitaria e regularizar grades lat/lon nao uniformes). O resultado e
    cacheado em disco por conteudo do arquivo de origem.
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        project_root = Path(__file__).resolve().parents[2]
        # O diretorio so e criado quando ha cache para gravar (ver
        # _prepare_remo_path): instanciar o adapter nao deve mexer em disco.
        self.cache_dir = cache_dir or (project_root / "data" / "cache" / "forcing")
        self._memory_cache: dict[tuple[str, str, str], Path] = {}

    def prepare_path(self, dataset_path: Path, forcing_source: str, dataset_kind: str) -> Path:
        source = (forcing_source or "COPERNICUS").strip().upper()
        kind = dataset_kind.strip().lower()
        path = Path(dataset_path)
        if source != "REMO":
            return path

        cache_key = (str(path.resolve()), source, kind)
        memoized = self._memory_cache.get(cache_key)
        if memoized is not None and memoized.exists():
            return memoized

        adapted_path = self._prepare_remo_path(path, kind)
        self._memory_cache[cache_key] = adapted_path
        return adapted_path

    def _prepare_remo_path(self, dataset_path: Path, dataset_kind: str) -> Path:
        if dataset_kind not in {"current", "wind"}:
            return dataset_path
        if not dataset_path.exists():
            raise FileNotFoundError(f"Forcing dataset not found: {dataset_path}")

        file_stat = dataset_path.stat()
        cache_token = "|".join(
            [
                str(dataset_path.resolve()),
                str(file_stat.st_mtime_ns),
                str(file_stat.st_size),
                dataset_kind,
            ]
        )
        digest = hashlib.sha1(cache_token.encode("utf-8")).hexdigest()[:16]
        target = self.cache_dir / f"{dataset_path.stem}.{dataset_kind}.remo.{digest}.nc"
        if target.exists():
            return target

        with xr.open_dataset(dataset_path) as ds:
            adapted = self._adapt_remo_dataset(ds, dataset_kind)
            if adapted is None:
                return dataset_path
            adapted.load()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = target.with_suffix(".tmp.nc")
            adapted.to_netcdf(tmp_path)
            tmp_path.replace(target)
            adapted.close()
        return target

    def _adapt_remo_dataset(self, dataset: xr.Dataset, dataset_kind: str) -> Optional[xr.Dataset]:
        ds = dataset.copy()
        changed = False

        rename_map: dict[str, str] = {}
        if "time1" in ds.dims and "time" not in ds.dims:
            rename_map["time1"] = "time"
        if rename_map:
            ds = ds.rename(rename_map)
            changed = True

        if "depth" in ds.dims and ds.sizes.get("depth", 0) == 1:
            ds = ds.squeeze(dim="depth", drop=True)
            changed = True
        elif "depth" in ds.coords and ds["depth"].size == 1:
            ds = ds.drop_vars("depth")
            changed = True

        if dataset_kind == "current":
            regridded = self._regularize_lat_lon_grid(ds)
            if regridded is not ds:
                ds = regridded
                changed = True

        if not changed:
            ds.close()
            return None
        return ds

    @staticmethod
    def _is_uniform_1d(values: np.ndarray) -> bool:
        if values.size < 3:
            return True
        diffs = np.diff(values.astype(float))
        if diffs.size == 0:
            return True
        tolerance = max(1e-8, abs(float(diffs[0])) * 1e-3)
        return bool(np.allclose(diffs, diffs[0], rtol=0.0, atol=tolerance))

    def _regularize_lat_lon_grid(self, dataset: xr.Dataset) -> xr.Dataset:
        lat_name = self._find_coord_name(dataset, ("latitude", "lat"))
        lon_name = self._find_coord_name(dataset, ("longitude", "lon"))
        if lat_name is None or lon_name is None:
            return dataset
        if dataset[lat_name].ndim != 1 or dataset[lon_name].ndim != 1:
            return dataset

        lat_values = np.asarray(dataset[lat_name].values, dtype=float)
        lon_values = np.asarray(dataset[lon_name].values, dtype=float)

        target_lat = lat_values
        target_lon = lon_values
        needs_lat = lat_values.size > 2 and not self._is_uniform_1d(lat_values)
        needs_lon = lon_values.size > 2 and not self._is_uniform_1d(lon_values)
        if needs_lat:
            target_lat = np.linspace(float(lat_values[0]), float(lat_values[-1]), lat_values.size)
        if needs_lon:
            target_lon = np.linspace(float(lon_values[0]), float(lon_values[-1]), lon_values.size)
        if not needs_lat and not needs_lon:
            return dataset

        interp_indexers = {}
        if needs_lat:
            interp_indexers[lat_name] = target_lat
        if needs_lon:
            interp_indexers[lon_name] = target_lon
        return dataset.interp(interp_indexers)

    @staticmethod
    def _find_coord_name(dataset: xr.Dataset, candidates: tuple[str, ...]) -> Optional[str]:
        for candidate in candidates:
            if candidate in dataset.coords or candidate in dataset.variables:
                return candidate
        return None


# --------------------------------------------------------------------------
# Download (Copernicus Marine)
# --------------------------------------------------------------------------


class CopernicusGateway:
    """Baixa do Copernicus Marine os datasets declarados no YAML do ambiente."""

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        login_file: Optional[Path] = None,
    ) -> None:
        self._username = username
        self._password = password
        self._login_file = Path(login_file) if login_file else None

    def _resolve_credentials(self) -> tuple[str, str]:
        if self._username and self._password:
            return self._username, self._password

        env_user = os.getenv("COPERNICUS_USERNAME")
        env_pwd = os.getenv("COPERNICUS_PASSWORD")
        if env_user and env_pwd:
            return env_user, env_pwd

        user_names = [
            value
            for value in [os.getenv("USERNAME"), os.getenv("username"), Path.home().name]
            if value
        ]
        user_names = list(dict.fromkeys(user_names))

        env_loginfile = os.getenv("COPERNICUS_LOGIN_FILE")
        login_candidates: list[Path] = []
        if self._login_file:
            login_candidates.append(self._login_file)
        if env_loginfile:
            login_candidates.append(Path(env_loginfile))
        project_root = Path(__file__).resolve().parents[2]
        for user_name in user_names:
            login_candidates.append(project_root / f"copernicus_login_{user_name}.json")
            login_candidates.append(Path.home() / f"copernicus_login_{user_name}.json")
        login_candidates.append(project_root / "copernicus_login.json")
        login_candidates.append(Path.home() / "copernicus_login.json")
        login_candidates = list(dict.fromkeys(login_candidates))

        login_path = next((path for path in login_candidates if path.exists()), None)
        if login_path is None:
            searched = "\n".join(f"- {path}" for path in login_candidates)
            raise FileNotFoundError(
                "Copernicus login file not found. Searched:\n" + searched
            )

        payload = json.loads(login_path.read_text(encoding="utf-8"))
        username = payload.get("user")
        password = payload.get("pwd")
        if not username or not password:
            raise ValueError(f"Invalid Copernicus login file: {login_path}")
        return username, password

    @staticmethod
    def _dataset_specs(config) -> list[dict]:
        specifics = config.copernicusmarine.specificities
        return [
            {
                "name": "sal_temp",
                "dataset_id": specifics.sal_temp_dataset_id,
                "dataset_path": Path(specifics.sal_temp_dataset_path),
                "variables": ["so", "thetao"],
            },
        ]

    def download_environment_data(
        self,
        config,
        force: bool = False,
        log_callback: Optional[Callable[[str], None]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> dict:
        def log(message: str) -> None:
            if log_callback is not None:
                log_callback(message)

        if (username and not password) or (password and not username):
            raise ValueError("Provide both Copernicus username and password.")
        if not username and not password:
            username, password = self._resolve_credentials()
        log("Trying Copernicus login...")
        if not cm.login(username=username, password=password, force_overwrite=True):
            raise RuntimeError("Copernicus login failed.")
        log("Copernicus login successful.")

        outcomes: list[dict] = []
        specs = self._dataset_specs(config)
        total = len(specs)
        min_lon = float(config.copernicusmarine.min_long)
        max_lon = float(config.copernicusmarine.max_long)
        min_lat = float(config.copernicusmarine.min_lat)
        max_lat = float(config.copernicusmarine.max_lat)
        start_dt = str(config.copernicusmarine.specificities.start_datetime)
        end_dt = str(config.copernicusmarine.specificities.end_datetime)
        for index, spec in enumerate(specs, start=1):
            dataset_path: Path = spec["dataset_path"]
            dataset_path.parent.mkdir(parents=True, exist_ok=True)

            if dataset_path.exists() and not force:
                log(f"{index}/{total} {spec['name']}: file already exists, skipping.")
                outcomes.append(
                    {
                        "name": spec["name"],
                        "path": str(dataset_path),
                        "status": "skipped",
                    }
                )
                continue

            log(f"{index}/{total} {spec['name']}: downloading...")
            log(
                f"{index}/{total} {spec['name']}: "
                f"dataset_id={spec['dataset_id']}, "
                f"lon=[{min_lon},{max_lon}], lat=[{min_lat},{max_lat}], "
                f"time=[{start_dt},{end_dt}]"
            )
            try:
                cm.subset(
                    dataset_id=spec["dataset_id"],
                    minimum_longitude=min_lon,
                    maximum_longitude=max_lon,
                    minimum_latitude=min_lat,
                    maximum_latitude=max_lat,
                    start_datetime=start_dt,
                    end_datetime=end_dt,
                    output_directory=dataset_path.parent,
                    output_filename=dataset_path.name,
                    variables=spec["variables"],
                )
            except Exception as exc:
                error_text = str(exc).lower()
                if "overlap" in error_text and min_lon < 0 and max_lon < 0:
                    wrap_min_lon = min_lon + 360.0
                    wrap_max_lon = max_lon + 360.0
                    log(
                        f"{index}/{total} {spec['name']}: retrying with wrapped longitudes "
                        f"[{wrap_min_lon},{wrap_max_lon}]"
                    )
                    cm.subset(
                        dataset_id=spec["dataset_id"],
                        minimum_longitude=wrap_min_lon,
                        maximum_longitude=wrap_max_lon,
                        minimum_latitude=min_lat,
                        maximum_latitude=max_lat,
                        start_datetime=start_dt,
                        end_datetime=end_dt,
                        output_directory=dataset_path.parent,
                        output_filename=dataset_path.name,
                        variables=spec["variables"],
                    )
                else:
                    raise RuntimeError(
                        f"Failed subset for dataset '{spec['name']}' "
                        f"(id={spec['dataset_id']})."
                    ) from exc
            log(f"{index}/{total} {spec['name']}: done.")
            outcomes.append(
                {
                    "name": spec["name"],
                    "path": str(dataset_path),
                    "status": "downloaded",
                }
            )

        return {"datasets": outcomes, "force": force}
