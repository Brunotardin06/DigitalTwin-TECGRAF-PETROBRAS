"""Sorteio reproduzivel dos parametros de cada membro do ensemble."""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd

from src.inputs.config import (
    SampledParameterSet,
    StochasticParameterConfig,
    StochasticRunConfig,
    TemporalLagConfig,
)


class ParameterSampler:
    """Sorteia os parametros de cada membro de forma reproduzivel.

    A semente de cada membro e `seed_base + simulation_id`, entao o mesmo
    seed na UI reproduz exatamente o mesmo ensemble. Dentro de um membro os
    tres parametros saem do mesmo gerador, sempre na ordem cdf, wdf, tau.
    """

    UNIT_TO_SECONDS = {
        "seconds": 1.0,
        "second": 1.0,
        "s": 1.0,
        "minutes": 60.0,
        "minute": 60.0,
        "min": 60.0,
        "hours": 3600.0,
        "hour": 3600.0,
        "h": 3600.0,
    }

    VALID_GRANULARITIES = {"seconds", "minutes"}

    def validate_run_config(self, config: StochasticRunConfig) -> None:
        if config.n_simulations <= 0:
            raise ValueError("n_simulations must be > 0")
        if config.seed is not None and not isinstance(config.seed, int):
            raise ValueError("seed must be an integer or None")
        if config.number_of_workers <= 0:
            raise ValueError("number_of_workers must be > 0")
        if config.execution_mode != "deterministic_ensemble":
            raise ValueError("Only deterministic_ensemble execution is supported")

        self.validate_parameter_config(config.cdf, "cdf")
        self.validate_parameter_config(config.wdf, "wdf")
        self.validate_temporal_lag_config(config.temporal_lag)

        grid = config.grid
        if grid.lon_min >= grid.lon_max:
            raise ValueError("grid lon_min must be < lon_max")
        if grid.lat_min >= grid.lat_max:
            raise ValueError("grid lat_min must be < lat_max")
        if grid.spatial_resolution <= 0:
            raise ValueError("grid spatial_resolution must be > 0")
        if not grid.crs:
            raise ValueError("grid crs is required")

    def validate_parameter_config(self, config: StochasticParameterConfig, name: str) -> None:
        if config.distribution != "normal":
            raise ValueError(f"{name} distribution must be 'normal'")
        if config.std < 0:
            raise ValueError(f"{name} std must be >= 0")
        if config.min_value >= config.max_value:
            raise ValueError(f"{name} min_value must be < max_value")
        if config.enabled:
            if config.mean < config.min_value or config.mean > config.max_value:
                raise ValueError(f"{name} mean must be inside min/max")
        elif config.default_value is None:
            raise ValueError(f"{name} default_value is required when disabled")

    def validate_temporal_lag_config(self, config: TemporalLagConfig) -> None:
        if config.distribution != "normal":
            raise ValueError("temporal_lag distribution must be 'normal'")
        if config.std < 0:
            raise ValueError("temporal_lag std must be >= 0")
        if config.min_value >= config.max_value:
            raise ValueError("temporal_lag min_value must be < max_value")
        if config.input_unit.lower() not in self.UNIT_TO_SECONDS:
            raise ValueError("temporal_lag input_unit must be seconds, minutes, or hours")
        if config.rounding_granularity.lower() not in self.VALID_GRANULARITIES:
            raise ValueError("temporal_lag rounding_granularity must be seconds or minutes")
        if config.enabled and (config.mean < config.min_value or config.mean > config.max_value):
            raise ValueError("temporal_lag mean must be inside min/max")

    def sample(self, config: StochasticRunConfig) -> list[SampledParameterSet]:
        self.validate_run_config(config)

        base_seed = int(config.seed) if config.seed is not None else int(np.random.SeedSequence().entropy)
        samples: list[SampledParameterSet] = []
        for simulation_id in range(config.n_simulations):
            simulation_seed = base_seed + simulation_id
            rng = np.random.default_rng(simulation_seed)
            cdf = self._sample_scalar(config.cdf, rng)
            wdf = self._sample_scalar(config.wdf, rng)
            lag_original = self._sample_temporal_lag_original(config.temporal_lag, rng)
            lag_seconds = self._temporal_lag_to_seconds(lag_original, config.temporal_lag)
            samples.append(
                SampledParameterSet(
                    simulation_id=simulation_id,
                    seed=simulation_seed,
                    cdf=float(cdf),
                    wdf=float(wdf),
                    temporal_lag_original_value=float(lag_original),
                    temporal_lag_input_unit=config.temporal_lag.input_unit,
                    temporal_lag_rounding_granularity=config.temporal_lag.rounding_granularity,
                    temporal_lag_seconds=float(lag_seconds),
                )
            )
        return samples

    @staticmethod
    def to_dataframe(samples: Iterable[SampledParameterSet]) -> pd.DataFrame:
        return pd.DataFrame([asdict(sample) for sample in samples])

    def _sample_scalar(self, config: StochasticParameterConfig, rng: np.random.Generator) -> float:
        if not config.enabled:
            return float(config.default_value)
        return self._sample_normal_bounded(
            rng=rng,
            mean=config.mean,
            std=config.std,
            min_value=config.min_value,
            max_value=config.max_value,
        )

    def _sample_temporal_lag_original(
        self,
        config: TemporalLagConfig,
        rng: np.random.Generator,
    ) -> float:
        if not config.enabled:
            unit_factor = self.UNIT_TO_SECONDS[config.input_unit.lower()]
            return float(config.default_seconds) / unit_factor
        return self._sample_normal_bounded(
            rng=rng,
            mean=config.mean,
            std=config.std,
            min_value=config.min_value,
            max_value=config.max_value,
        )

    def _temporal_lag_to_seconds(self, value: float, config: TemporalLagConfig) -> float:
        seconds = float(value) * self.UNIT_TO_SECONDS[config.input_unit.lower()]
        if config.rounding_granularity.lower() == "seconds":
            return float(round(seconds))
        if config.rounding_granularity.lower() == "minutes":
            return float(round(seconds / 60.0) * 60.0)
        raise ValueError("temporal_lag rounding_granularity must be seconds or minutes")

    @staticmethod
    def _sample_normal_bounded(
        rng: np.random.Generator,
        mean: float,
        std: float,
        min_value: float,
        max_value: float,
    ) -> float:
        if std == 0:
            return float(np.clip(mean, min_value, max_value))

        # Normal truncada por rejeicao: preserva a forma da distribuicao dentro
        # do intervalo, ao contrario de simplesmente clipar (que acumularia
        # massa nas bordas). O clip final so protege contra min/max muito
        # estreitos em relacao ao desvio, onde a rejeicao nao converge.
        for _ in range(10_000):
            value = float(rng.normal(mean, std))
            if min_value <= value <= max_value:
                return value

        value = rng.normal(mean, std)
        return float(np.clip(value, min_value, max_value))
