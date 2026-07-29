"""Execucao de uma simulacao de deriva no OpenDrift, do seed ao arquivo .nc."""

import os
from datetime import timedelta
from pathlib import Path
from typing import Optional

import geopandas as gpd
from opendrift.models.openoil import OpenOil

from src.inputs import forcing
from src.inputs.forcing import ForcingDatasetAdapter
from src.inputs.spills import SpillRepository, generate_random_points_in_polygon

DEFAULT_WIND_DRIFT_FACTOR = 0.015
DEFAULT_OIL_TYPE = "SOCKEYE SWEET"


class SimulationService:
    """Roda simulacoes de deriva de oleo no OpenDrift.

    Dois modos de semeadura, que diferem em de onde vem a janela temporal:

      * simulate_drift        semeia no poligono da primeira mancha observada;
                              a janela sai das proprias manchas.
      * simulate_point_drift  semeia num ponto lon/lat informado; a data e a
                              duracao sao parametros diretos.
    """

    def __init__(
        self,
        spill_repository: Optional[SpillRepository] = None,
        forcing_dataset_adapter: Optional[ForcingDatasetAdapter] = None,
    ) -> None:
        self.spill_repository = spill_repository or SpillRepository()
        self.forcing_dataset_adapter = forcing_dataset_adapter or ForcingDatasetAdapter()

    def simulate_point_drift(
        self,
        config,
        lon: float,
        lat: float,
        seed_time_start,
        duration_days: float,
        out_filename: str,
        leak_duration_hours: float = 0.0,
        seed_radius_m: float = 1000.0,
        num_seed_elements: Optional[int] = None,
        oil_type: Optional[str] = None,
        wind_drift_factor: Optional[float] = None,
        current_drift_factor: Optional[float] = None,
        processes_dispersion: Optional[bool] = None,
        processes_evaporation: Optional[bool] = None,
        time_step_minutes: Optional[float] = None,
        output_time_step_minutes: Optional[float] = None,
        forcing_source: str = "COPERNICUS",
        current_dataset_path=None,
        wind_dataset_path=None,
        current_dataset_paths=None,
        wind_dataset_paths=None,
        environmental_offset_hours: Optional[float] = None,
        use_sal_temp: bool = True,
        skip_animation: bool = True,
    ) -> Path:
        """Simula a deriva a partir de um ponto, data e duracao informados.

        Diferente de simulate_drift, nao depende de manchas observadas: serve
        para prever o destino de um vazamento hipotetico em qualquer data
        coberta pelos dados de forcing.

        Retorna o caminho do .nc gerado.
        """
        current_paths, wind_paths, sal_temp_path = forcing.resolve_forcing_paths(
            config=config,
            forcing_source=forcing_source,
            adapter=self.forcing_dataset_adapter,
            current_dataset_path=current_dataset_path,
            wind_dataset_path=wind_dataset_path,
            current_dataset_paths=current_dataset_paths,
            wind_dataset_paths=wind_dataset_paths,
        )
        # Sal/temp so entra se existir em disco: sem ele o OpenOil ainda roda,
        # e exigi-lo impediria simular em datas fora do dataset baixado.
        if not use_sal_temp or not Path(sal_temp_path).exists():
            sal_temp_path = None

        model = OpenOil(loglevel=50)
        model.add_reader(
            forcing.build_readers(
                current_paths=current_paths,
                wind_paths=wind_paths,
                sal_temp_path=sal_temp_path,
                environmental_offset_hours=environmental_offset_hours,
            )
        )
        # Sem reader de vento o OpenDrift abortaria por variavel ausente; o
        # fallback zero deixa a deriva ser puramente por corrente.
        if not wind_paths:
            model.set_config("environment:fallback:x_wind", 0)
            model.set_config("environment:fallback:y_wind", 0)

        model.set_config("drift:advection_scheme", "runge-kutta4")
        model.set_config("drift:stokes_drift", False)

        wdf = wind_drift_factor
        if wdf is None:
            wdf = getattr(config.simulation, "wind_drift_factor", DEFAULT_WIND_DRIFT_FACTOR)
        model.set_config("seed:wind_drift_factor", wdf)
        if current_drift_factor is not None:
            model.set_config("seed:current_drift_factor", float(current_drift_factor))
        if processes_dispersion is not None:
            model.set_config("processes:dispersion", bool(processes_dispersion))
        if processes_evaporation is not None:
            model.set_config("processes:evaporation", bool(processes_evaporation))

        # Vazamento instantaneo (duracao 0) vira um instante unico; com duracao
        # o OpenDrift distribui as particulas ao longo do intervalo.
        seed_time_end = seed_time_start + timedelta(hours=float(leak_duration_hours or 0.0))
        seed_time = (
            seed_time_start
            if seed_time_end == seed_time_start
            else [seed_time_start, seed_time_end]
        )
        model.seed_elements(
            lon=float(lon),
            lat=float(lat),
            time=seed_time,
            number=int(num_seed_elements or config.simulation.num_seed_elements),
            radius=float(seed_radius_m),
            oil_type=oil_type or getattr(config.simulation, "oil_type", DEFAULT_OIL_TYPE),
        )

        duration = timedelta(days=float(duration_days))
        step = float(time_step_minutes or config.simulation.time_step_minutes)
        step_output = float(
            output_time_step_minutes or config.simulation.output_time_step_minutes
        )

        output_dir = Path(config.paths.simulation_data) / config.simulation.name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{out_filename}.nc"

        print(f"Seeding {num_seed_elements} elements at ({lon}, {lat}) on {seed_time_start}")
        model.run(
            duration=duration,
            time_step=step * 60,
            time_step_output=step_output * 60,
            outfile=str(output_path.absolute()),
        )
        print(f"Simulation {out_filename} completed successfully.")

        if not skip_animation:
            model.animation(
                filename=str((output_dir / f"{out_filename}.gif").absolute()),
                background=["x_sea_water_velocity", "y_sea_water_velocity"],
                vmin=-1,
                vmax=1,
                fast=True,
                fps=6,
            )
        return output_path

    def simulate_drift(
        self,
        manchas,
        out_filename,
        config,
        skip_animation,
        padding_animation_frame,
        wind_drift_factor=None,
        current_drift_factor=None,
        oil_type=None,
        processes_dispersion=None,
        processes_evaporation=None,
        forcing_source="COPERNICUS",
        current_dataset_path=None,
        wind_dataset_path=None,
        current_dataset_paths=None,
        wind_dataset_paths=None,
        observed_offset_hours=None,
        environmental_offset_hours=None,
        temporal_lag_seconds=None,
    ):
        # Janela temporal e ponto de partida saem das manchas observadas.
        offset_hours = observed_offset_hours
        if offset_hours is None:
            offset_hours = float(
                getattr(config.copernicusmarine.specificities, "datetime_offset_hours", 0) or 0.0
            )
        self.spill_repository.ensure_datetime_column(manchas, offset_hours=offset_hours)
        manchas.sort_values("datetime", inplace=True)

        seed_time_start = manchas["datetime"].iloc[0]
        end_time = manchas["datetime"].max()
        initial_subset = manchas[manchas["datetime"] == seed_time_start]
        mancha_inicial_geo = (
            initial_subset.geometry.union_all()
            if hasattr(initial_subset.geometry, "union_all")
            else initial_subset.geometry.unary_union
        )

        print("First arrival datetime:", seed_time_start)
        print("Most recent spill datetime:", end_time)

        # As particulas nascem espalhadas dentro da primeira mancha observada.
        points = generate_random_points_in_polygon(
            mancha_inicial_geo, config.simulation.num_seed_elements
        )
        elements = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")

        current_paths, wind_paths, sal_temp_path = forcing.resolve_forcing_paths(
            config=config,
            forcing_source=forcing_source,
            adapter=self.forcing_dataset_adapter,
            current_dataset_path=current_dataset_path,
            wind_dataset_path=wind_dataset_path,
            current_dataset_paths=current_dataset_paths,
            wind_dataset_paths=wind_dataset_paths,
        )

        model = OpenOil(loglevel=50)
        model.add_reader(
            forcing.build_readers(
                current_paths=current_paths,
                wind_paths=wind_paths,
                sal_temp_path=sal_temp_path,
                environmental_offset_hours=environmental_offset_hours,
                temporal_lag_seconds=temporal_lag_seconds,
            )
        )

        minlon, minlat, maxlon, maxlat = manchas.total_bounds
        padding_lon = (maxlon - minlon) * padding_animation_frame
        padding_lat = (maxlat - minlat) * padding_animation_frame
        min_lon = minlon - padding_lon
        max_lon = maxlon + padding_lon
        min_lat = minlat - padding_lat
        max_lat = maxlat + padding_lat

        print("Seeding elements...")
        model.set_config("drift:advection_scheme", "runge-kutta4")
        # Ondas ficam desligadas em todo o projeto: os datasets de forcing
        # disponiveis nao trazem campo de onda.
        model.set_config("drift:stokes_drift", False)

        wdf = wind_drift_factor
        if wdf is None:
            wdf = getattr(config.simulation, "wind_drift_factor", DEFAULT_WIND_DRIFT_FACTOR)
        model.set_config("seed:wind_drift_factor", wdf)

        if current_drift_factor is None:
            current_drift_factor = getattr(config.simulation, "current_drift_factor", None)
        if current_drift_factor is not None:
            model.set_config("seed:current_drift_factor", float(current_drift_factor))
        if processes_dispersion is not None:
            model.set_config("processes:dispersion", bool(processes_dispersion))
        if processes_evaporation is not None:
            model.set_config("processes:evaporation", bool(processes_evaporation))

        selected_oil = oil_type or getattr(config.simulation, "oil_type", DEFAULT_OIL_TYPE)
        model.seed_from_geopandas(
            geodataframe=elements,
            time=seed_time_start,
            oil_type=selected_oil,
        )

        if os.path.exists(Path(config.paths.simulation_data) / config.simulation.name):
            output_filename = (
                Path(config.paths.simulation_data)
                / config.simulation.name
                / f"{out_filename}.nc"
            )
            print("Running...")
            model.prepare_run()
            model.run(
                end_time=end_time,
                time_step=config.simulation.time_step_minutes * 60,
                time_step_output=config.simulation.output_time_step_minutes * 60,
                outfile=str(output_filename.absolute()),
            )
            model.elements
        else:
            print("Erro o.run(): o path do arquivo de saída provavelmente não é correto.")

        print(f"Simulation {out_filename} completed successfully.")
        if skip_animation:
            return model

        animation_filename = (
            Path(config.paths.simulation_data)
            / config.simulation.name
            / f"{out_filename}.gif"
        )
        model.animation(
            filename=str(animation_filename.absolute()),
            background=["x_sea_water_velocity", "y_sea_water_velocity"],
            corners=[min_lon, max_lon, min_lat, max_lat],
            vmin=-1,
            vmax=1,
            fast=True,
            fps=6,
        )
        return model
