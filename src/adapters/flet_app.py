from __future__ import annotations

import contextlib
import queue
import threading
import time
import traceback
from pathlib import Path

import flet as ft

from src.adapters.flet_ui import (
    ArtifactsViewBindings,
    ExecutionViewBindings,
    QueueWriter,
    ResultsViewBindings,
    SetupViewBindings,
    StochasticViewBindings,
    build_artifact_list,
    build_artifacts_view,
    build_execution_view,
    build_frame_list,
    extract_observed_bounds,
    build_results_view,
    build_run_id,
    build_setup_view,
    build_sidebar,
    build_stochastic_view,
    extract_metrics,
    list_environments,
    open_path,
    parse_float,
    stage_observed_zip,
    validate_observed_zip,
)
from src.application.dto import (
    StochasticValidationRunRequest,
    ValidationRunRequest,
    ValidationRunResult,
)
from src.application.simulation_controller import SimulationController
from src.services.stochastic.stochastic_models import (
    StochasticGridConfig,
    StochasticParameterConfig,
    StochasticRunConfig,
    TemporalLagConfig,
)

EMPTY_IMAGE_SRC = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///ywAAAAAAQABAAACAUwAOw=="
OBSERVED_BOUNDS_PADDING_DEG = 1.0
MIN_OBSERVED_BOUNDS_SPAN_DEG = 2.0
MAX_ENVIRONMENTAL_OFFSET_RANGE_HOURS = 24
STOCHASTIC_GRID_MARGIN_DEG = 1.0


def _enum_value(enum_name: str, member_name: str, fallback):
    enum_obj = getattr(ft, enum_name, None)
    if enum_obj is None:
        return fallback
    return getattr(enum_obj, member_name, fallback)


def _expand_bounds(
    bounds: tuple[float, float, float, float],
    padding_deg: float = OBSERVED_BOUNDS_PADDING_DEG,
    min_span_deg: float = MIN_OBSERVED_BOUNDS_SPAN_DEG,
) -> tuple[float, float, float, float]:
    min_lon, max_lon, min_lat, max_lat = bounds
    min_lon_p = min_lon - padding_deg
    max_lon_p = max_lon + padding_deg
    min_lat_p = min_lat - padding_deg
    max_lat_p = max_lat + padding_deg

    lon_span = max_lon_p - min_lon_p
    lat_span = max_lat_p - min_lat_p
    if lon_span < min_span_deg:
        lon_center = 0.5 * (min_lon_p + max_lon_p)
        half = 0.5 * min_span_deg
        min_lon_p = lon_center - half
        max_lon_p = lon_center + half
    if lat_span < min_span_deg:
        lat_center = 0.5 * (min_lat_p + max_lat_p)
        half = 0.5 * min_span_deg
        min_lat_p = lat_center - half
        max_lat_p = lat_center + half

    return (min_lon_p, max_lon_p, min_lat_p, max_lat_p)


def _parse_optional_float(value: str | None, field_name: str) -> float | None:
    if value is None or not str(value).strip():
        return None
    return parse_float(str(value), field_name)


def main(page: ft.Page) -> None:
    page.title = "Oil Spill Drift"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.window.min_width = 1200
    page.window.min_height = 760
    page.padding = 0
    page.bgcolor = "#D8DEE6"
    page.scroll = _enum_value("ScrollMode", "AUTO", "auto")

    project_root = Path(__file__).resolve().parents[2]
    environment_names = list_environments(project_root)

    state = {
        "selected_zip": None,
        "selected_current_datasets": [],
        "selected_wind_datasets": [],
        "staged_zip": None,
        "run_id": None,
        "running": False,
        "downloading": False,
        "cancel_event": threading.Event(),
        "out_dir": None,
        "sim_path": None,
        "frames": [],
        "start_time": None,
        "next_progress_log_pct": 0,
    }
    event_queue: queue.Queue = queue.Queue()

    def show_message(message: str, *, error: bool = False) -> None:
        snack = ft.SnackBar(
            content=ft.Text(message),
            bgcolor="#B91C1C" if error else "#1E3A8A",
        )
        # Flet classico (<=0.28) usa page.open(); o atributo page.snack_bar
        # foi removido. Mantem fallback para versoes antigas.
        if hasattr(page, "open"):
            page.open(snack)
        else:
            page.snack_bar = snack
            page.snack_bar.open = True
            page.update()

    selected_zip_text = ft.Text("Nenhum arquivo selecionado", color="#4B6385")
    selected_current_dataset_text = ft.Text(
        "Nenhum arquivo selecionado (usa water_dataset_path do YAML).",
        color="#4B6385",
    )
    selected_wind_dataset_text = ft.Text(
        "Nenhum arquivo selecionado (reader de vento opcional).",
        color="#4B6385",
    )
    environment_download_status_text = ft.Text("Dados não baixados nesta sessão.", color="#4B6385")
    forcing_source_dropdown = ft.Dropdown(
        label="Forcing Source",
        value="COPERNICUS",
        options=[
            ft.dropdown.Option("COPERNICUS"),
            ft.dropdown.Option("NOAA"),
            ft.dropdown.Option("REMO"),
        ],
        width=220,
    )

    status_title = ft.Text("Idle", size=28, weight=ft.FontWeight.W_700, color="#0F172A")
    status_subtitle = ft.Text("Aguardando execução", size=16, color="#4B6385")
    progress_text = ft.Text("0%", color="#4B6385")
    progress_bar = ft.ProgressBar(value=0, bgcolor="#C9CED6", color="#030523", height=10)
    log_view = ft.ListView(expand=True, spacing=4, auto_scroll=True, height=260)

    result_score_text = ft.Text("N/A", size=34, weight=ft.FontWeight.W_700, color="#0F172A")
    result_runtime_text = ft.Text("N/A", size=26, weight=ft.FontWeight.W_600, color="#0F172A")
    best_wdf_text = ft.Text("N/A", size=24, weight=ft.FontWeight.W_600)
    best_cdf_text = ft.Text("N/A", size=24, weight=ft.FontWeight.W_600)
    best_environmental_offset_text = ft.Text("N/A", size=24, weight=ft.FontWeight.W_600)

    frame_image = ft.Image(
        src=EMPTY_IMAGE_SRC,
        fit=_enum_value("ImageFit", "CONTAIN", "contain"),
        width=980,
        height=420,
        border_radius=12,
        visible=False,
    )
    frame_label = ft.Text("Sem imagens de comparação ainda.", color="#4B6385")

    def on_frame_slider_change(e: ft.ControlEvent) -> None:
        if not state["frames"]:
            return
        idx = int(round(e.control.value))
        idx = max(0, min(idx, len(state["frames"]) - 1))
        frame_path = state["frames"][idx]
        frame_image.src = str(frame_path)
        frame_label.value = f"Step {idx + 1}/{len(state['frames'])}: {frame_path.name}"
        page.update()

    frame_slider = ft.Slider(
        min=0,
        max=0,
        divisions=1,
        disabled=True,
        on_change=on_frame_slider_change,
    )

    output_path_text = ft.Text("N/A", color="#4B6385")
    artifacts_column = ft.Column(
        spacing=8,
        scroll=_enum_value("ScrollMode", "AUTO", "auto"),
        height=340,
    )

    environment_dropdown = ft.Dropdown(
        label="Environment",
        value=environment_names[0],
        options=[ft.dropdown.Option(name) for name in environment_names],
        width=320,
    )
    start_index_field = ft.TextField(label="Start timestep index", value="1", width=180)
    environmental_offset_hours_field = ft.TextField(
        label="Environmental offset range (h)",
        value="10",
        width=180,
    )
    copernicus_username_field = ft.TextField(
        label="Copernicus username",
        width=320,
    )
    copernicus_password_field = ft.TextField(
        label="Copernicus password",
        password=True,
        can_reveal_password=True,
        width=320,
    )

    wdf_min_field = ft.TextField(label="WDF min", value="0.015", width=140)
    wdf_max_field = ft.TextField(label="WDF max", value="0.040", width=140)
    wdf_step_field = ft.TextField(label="WDF step", value="0.0025", width=140)
    fixed_wdf_field = ft.TextField(label="Fixed WDF", value="0.035", width=140)

    cdf_min_field = ft.TextField(label="CDF min", value="0.5", width=140)
    cdf_max_field = ft.TextField(label="CDF max", value="1.5", width=140)
    cdf_step_field = ft.TextField(label="CDF step", value="0.1", width=140)
    fixed_cdf_field = ft.TextField(label="Fixed CDF", value="1.0", width=140)

    stochastic_selected_zip_text = ft.Text("Nenhum arquivo selecionado", color="#4B6385")
    stochastic_selected_current_dataset_text = ft.Text(
        "Nenhum arquivo selecionado (usa water_dataset_path do YAML).",
        color="#4B6385",
    )
    stochastic_selected_wind_dataset_text = ft.Text(
        "Nenhum arquivo selecionado (reader de vento opcional).",
        color="#4B6385",
    )
    stochastic_forcing_source_dropdown = ft.Dropdown(
        label="Forcing Source",
        value="COPERNICUS",
        options=[
            ft.dropdown.Option("COPERNICUS"),
            ft.dropdown.Option("NOAA"),
            ft.dropdown.Option("REMO"),
        ],
        width=220,
    )
    stochastic_environment_dropdown = ft.Dropdown(
        label="Environment",
        value=environment_names[0],
        options=[ft.dropdown.Option(name) for name in environment_names],
        width=260,
    )
    stochastic_start_index_field = ft.TextField(label="Start timestep index", value="1", width=180)
    stochastic_base_environmental_offset_hours_field = ft.TextField(
        label="Base env offset (h)",
        value="-3",
        width=180,
    )
    stochastic_run_name_field = ft.TextField(label="Run name", value="", width=240)
    stochastic_n_simulations_field = ft.TextField(label="N simulations", value="10", width=160)
    stochastic_number_of_workers_field = ft.TextField(label="Number of workers", value="4", width=180)
    stochastic_seed_field = ft.TextField(label="Seed", value="42", width=120)

    stochastic_cdf_enabled_checkbox = ft.Checkbox(label="Vary CDF", value=True)
    stochastic_cdf_mean_field = ft.TextField(label="CDF mean", value="1.0", width=130)
    stochastic_cdf_std_field = ft.TextField(label="CDF std", value="0.1", width=130)
    stochastic_cdf_min_field = ft.TextField(label="CDF min", value="0.8", width=130)
    stochastic_cdf_max_field = ft.TextField(label="CDF max", value="1.2", width=130)

    stochastic_wdf_enabled_checkbox = ft.Checkbox(label="Vary WDF", value=True)
    stochastic_wdf_mean_field = ft.TextField(label="WDF mean", value="0.035", width=130)
    stochastic_wdf_std_field = ft.TextField(label="WDF std", value="0.001", width=130)
    stochastic_wdf_min_field = ft.TextField(label="WDF min", value="0.032", width=130)
    stochastic_wdf_max_field = ft.TextField(label="WDF max", value="0.038", width=130)

    stochastic_tau_enabled_checkbox = ft.Checkbox(label="Vary temporal lag", value=True)
    stochastic_tau_mean_field = ft.TextField(label="Tau mean", value="0", width=130)
    stochastic_tau_std_field = ft.TextField(label="Tau std", value="30", width=130)
    stochastic_tau_min_field = ft.TextField(label="Tau min", value="-120", width=130)
    stochastic_tau_max_field = ft.TextField(label="Tau max", value="120", width=130)
    stochastic_tau_input_unit_dropdown = ft.Dropdown(
        label="Tau unit",
        value="minutes",
        options=[
            ft.dropdown.Option("seconds"),
            ft.dropdown.Option("minutes"),
            ft.dropdown.Option("hours"),
        ],
        width=140,
    )
    stochastic_tau_rounding_dropdown = ft.Dropdown(
        label="Granularity",
        value="minutes",
        options=[
            ft.dropdown.Option("seconds"),
            ft.dropdown.Option("minutes"),
        ],
        width=150,
    )

    stochastic_grid_lon_min_field = ft.TextField(label="Lon min", value="", width=130)
    stochastic_grid_lon_max_field = ft.TextField(label="Lon max", value="", width=130)
    stochastic_grid_lat_min_field = ft.TextField(label="Lat min", value="", width=130)
    stochastic_grid_lat_max_field = ft.TextField(label="Lat max", value="", width=130)
    stochastic_grid_resolution_field = ft.TextField(label="Resolution", value="0.001", width=140)
    stochastic_grid_margin_field = ft.TextField(
        label="Margin",
        value=f"{STOCHASTIC_GRID_MARGIN_DEG:g}",
        width=120,
    )

    run_mode_dropdown = ft.Dropdown(
        label="Mode",
        value="OPTIMIZATION",
        options=[
            ft.dropdown.Option("OPTIMIZATION"),
            ft.dropdown.Option("NO OPTIMIZATION"),
        ],
        width=220,
    )

    def build_offset_values() -> list[int]:
        raw_value = parse_float(
            environmental_offset_hours_field.value,
            "Environmental offset range (h)",
        )
        if not raw_value.is_integer():
            raise ValueError("Environmental offset range (h) deve ser um numero inteiro.")
        max_offset = int(abs(raw_value))
        if max_offset > MAX_ENVIRONMENTAL_OFFSET_RANGE_HOURS:
            raise ValueError(
                "Environmental offset range (h) deve estar entre "
                f"0 e {MAX_ENVIRONMENTAL_OFFSET_RANGE_HOURS}."
            )
        return list(range(-max_offset, max_offset + 1))

    def offset_suffix(offset_hours: int) -> str:
        if offset_hours < 0:
            return f"envm{abs(offset_hours):02d}"
        if offset_hours > 0:
            return f"envp{offset_hours:02d}"
        return "envp00"

    def append_log(message: str) -> None:
        log_view.controls.append(ft.Text(message, size=13, color="#0F172A"))
        if len(log_view.controls) > 1000:
            log_view.controls = log_view.controls[-1000:]

    def reset_execution_panel() -> None:
        status_title.value = "Running"
        status_subtitle.value = "Executando validação..."
        progress_bar.value = 0
        progress_text.value = "0%"
        log_view.controls.clear()
        state["next_progress_log_pct"] = 0

    def refresh_artifacts() -> None:
        artifacts_column.controls.clear()
        out_dir = state["out_dir"]
        if out_dir is None:
            page.update()
            return
        for file_path in build_artifact_list(out_dir):
            artifacts_column.controls.append(
                ft.Container(
                    bgcolor="#EEF2F7",
                    border=ft.border.all(1, "#CBD5E1"),
                    border_radius=10,
                    padding=12,
                    content=ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        controls=[
                            ft.Column(
                                spacing=2,
                                controls=[
                                    ft.Text(file_path.name, weight=ft.FontWeight.W_600),
                                    ft.Text(
                                        f"{file_path.stat().st_size / 1024:.1f} KB",
                                        size=12,
                                        color="#4B6385",
                                    ),
                                ],
                            ),
                            ft.OutlinedButton(
                                "Open",
                                on_click=lambda e, p=file_path: open_path(p),
                            ),
                        ],
                    ),
                )
            )

    def refresh_results() -> None:
        out_dir = state["out_dir"]
        sim_path = state["sim_path"]
        if out_dir is None:
            return

        metrics = extract_metrics(out_dir)
        if metrics["best_skillscore"] is None:
            result_score_text.value = "N/A"
        else:
            result_score_text.value = f"{metrics['best_skillscore']:.3f}"
        best_wdf_text.value = (
            f"{metrics['best_wdf']:.4f}" if metrics["best_wdf"] is not None else "N/A"
        )
        best_cdf_text.value = (
            f"{metrics['best_cdf']:.2f}" if metrics["best_cdf"] is not None else "N/A"
        )
        best_environmental_offset_text.value = (
            f"{metrics['best_environmental_offset']:+g} h"
            if metrics["best_environmental_offset"] is not None
            else "N/A"
        )
        state["frames"] = build_frame_list(out_dir, sim_path)
        if state["frames"]:
            frame_slider.disabled = False
            frame_slider.min = 0
            frame_slider.max = len(state["frames"]) - 1
            frame_slider.divisions = max(1, len(state["frames"]) - 1)
            frame_slider.value = 0
            frame_image.src = str(state["frames"][0])
            frame_image.visible = True
            frame_label.value = f"Step 1/{len(state['frames'])}: {state['frames'][0].name}"
        else:
            frame_slider.disabled = True
            frame_slider.min = 0
            frame_slider.max = 0
            frame_slider.divisions = 1
            frame_slider.value = 0
            frame_image.src = EMPTY_IMAGE_SRC
            frame_image.visible = False
            frame_label.value = "Sem imagens de comparação ainda."

    def recover_output_from_run_id(run_id: str | None) -> tuple[Path | None, Path | None]:
        if not run_id:
            return None, None
        out_dir = project_root / "data" / "2-simulated" / run_id
        if not out_dir.exists():
            return None, None
        nc_files = sorted(out_dir.glob("*.nc"), key=lambda p: p.stat().st_mtime, reverse=True)
        sim_path = nc_files[0] if nc_files else None
        return out_dir, sim_path

    def handle_queue_event(payload: dict) -> None:
        event_type = payload.get("type")
        if event_type == "log":
            append_log(payload.get("message", ""))
        elif event_type == "env_download_log":
            append_log(payload.get("message", ""))
        elif event_type == "env_download_done":
            state["downloading"] = False
            environment = payload.get("environment", "")
            environment_download_status_text.value = (
                f"{environment}: download concluído."
            )
            append_log(f"Environment {environment}: sal_temp dataset ready.")
            show_message(f"Download do ambiente {environment} concluído.")
        elif event_type == "env_download_error":
            state["downloading"] = False
            environment = payload.get("environment", "")
            environment_download_status_text.value = f"{environment}: falha no download."
            append_log(payload.get("message", "Download failed."))
            show_message(
                f"Falha no download do ambiente {environment}. Veja os logs.",
                error=True,
            )
        elif event_type == "progress":
            done = int(payload.get("done", 0))
            total = max(1, int(payload.get("total", 1)))
            pct = int(round((done / total) * 100))
            progress_bar.value = done / total
            progress_text.value = f"{pct}%"
            status_subtitle.value = f"Otimização em andamento ({done}/{total})"
            next_pct = int(state.get("next_progress_log_pct", 0))
            if done == total or pct >= next_pct:
                append_log(f"Optimization progress: {done}/{total} ({pct}%)")
                if done == total:
                    state["next_progress_log_pct"] = 101
                else:
                    while next_pct <= pct:
                        next_pct += 5
                    state["next_progress_log_pct"] = next_pct
        elif event_type == "stochastic_progress":
            done = int(payload.get("done", 0))
            total = max(1, int(payload.get("total", 1)))
            pct = int(round((done / total) * 100))
            progress_bar.value = done / total
            progress_text.value = f"{pct}%"
            status_subtitle.value = f"Simulação estocástica em andamento ({done}/{total})"
            next_pct = int(state.get("next_progress_log_pct", 0))
            if done == total or pct >= next_pct:
                append_log(f"Stochastic progress: {done}/{total} ({pct}%)")
                if done == total:
                    state["next_progress_log_pct"] = 101
                else:
                    while next_pct <= pct:
                        next_pct += 5
                    state["next_progress_log_pct"] = next_pct
        elif event_type == "batch_start":
            index = int(payload.get("index", 0))
            total = int(payload.get("total", 0))
            offset = payload.get("offset")
            offset_values = payload.get("offset_values")
            run_id = payload.get("run_id", "")
            state["run_id"] = run_id
            state["next_progress_log_pct"] = 0
            progress_bar.value = 0
            progress_text.value = "0%"
            status_title.value = "Running"
            offset_label = None
            if offset_values:
                offset_label = f"otimizando {len(offset_values)} offsets"
            elif isinstance(offset, (int, float)):
                offset_label = f"offset ambiental {offset:+g} h"
            status_subtitle.value = (
                f"{offset_label} ({index}/{total})"
                if offset_label
                else f"Executando lote ({index}/{total})"
            )
            append_log(
                f"Starting {offset_label} ({index}/{total}) - Run ID: {run_id}"
                if offset_label
                else f"Starting batch run {index}/{total} - Run ID: {run_id}"
            )
        elif event_type == "batch_done":
            result: ValidationRunResult = payload["result"]
            offset = payload.get("offset")
            offset_values = payload.get("offset_values")
            index = int(payload.get("index", 0))
            total = int(payload.get("total", 0))
            state["out_dir"] = result.out_dir
            state["sim_path"] = result.sim_path
            output_path_text.value = str(result.out_dir)
            refresh_results()
            refresh_artifacts()
            offset_label = None
            if offset_values:
                offset_label = f"optimized {len(offset_values)} offsets"
            elif isinstance(offset, (int, float)):
                offset_label = f"offset {offset:+g} h"
            append_log(
                f"Finished {offset_label} ({index}/{total}): {result.out_dir}"
                if offset_label
                else f"Finished batch run {index}/{total}: {result.out_dir}"
            )
        elif event_type == "done":
            result: ValidationRunResult = payload["result"]
            state["running"] = False
            state["out_dir"] = result.out_dir
            state["sim_path"] = result.sim_path
            duration = time.monotonic() - (state["start_time"] or time.monotonic())
            result_runtime_text.value = f"{duration:.1f} s"
            status_title.value = "Completed"
            status_subtitle.value = "Execução finalizada com sucesso"
            progress_bar.value = 1.0
            progress_text.value = "100%"
            output_path_text.value = str(result.out_dir)
            refresh_results()
            refresh_artifacts()
            show_message("Execução concluída.")
        elif event_type == "stochastic_done":
            result = payload["result"]
            state["running"] = False
            state["out_dir"] = result.output_path
            state["sim_path"] = None
            duration = time.monotonic() - (state["start_time"] or time.monotonic())
            result_runtime_text.value = f"{duration:.1f} s"
            status_title.value = "Completed"
            status_subtitle.value = (
                f"Estocástica finalizada: "
                f"{result.successful_simulations}/{result.total_simulations} válidas"
            )
            progress_bar.value = 1.0
            progress_text.value = "100%"
            output_path_text.value = str(result.output_path)
            refresh_results()
            refresh_artifacts()
            show_message("Simulação estocástica concluída.")
        elif event_type == "cancelled":
            state["running"] = False
            recovered_out_dir, recovered_sim_path = recover_output_from_run_id(state.get("run_id"))
            if recovered_out_dir is not None:
                state["out_dir"] = recovered_out_dir
                state["sim_path"] = recovered_sim_path
                output_path_text.value = str(recovered_out_dir)
                status_title.value = "Completed"
                status_subtitle.value = "Execução finalizada com artefatos recuperados"
                append_log(
                    f"Recovered outputs from {recovered_out_dir} after late cancellation signal."
                )
                refresh_results()
                refresh_artifacts()
                show_message("Execução finalizada e resultados recuperados.")
            else:
                status_title.value = "Cancelled"
                status_subtitle.value = "Execução cancelada pelo usuário"
                show_message("Execução cancelada.")
        elif event_type == "error":
            state["running"] = False
            status_title.value = "Failed"
            status_subtitle.value = "Falha durante a execução"
            append_log(payload.get("message", "Unknown error"))
            show_message("Falha na execução. Veja os logs.", error=True)
        page.update()

    def event_consumer_loop() -> None:
        while True:
            payload = event_queue.get()
            handle_queue_event(payload)

    consumer_thread = threading.Thread(target=event_consumer_loop, daemon=True)
    consumer_thread.start()

    def worker_download_environment(
        environment: str,
        copernicus_username: str,
        copernicus_password: str,
        observed_bounds: tuple[float, float, float, float] | None,
        force_download: bool,
    ) -> None:
        controller = SimulationController()

        def on_log(message: str) -> None:
            event_queue.put({"type": "env_download_log", "message": message})

        try:
            controller.download_environment_data(
                environment=environment,
                config_name="main",
                force=force_download,
                log_callback=on_log,
                copernicus_username=copernicus_username,
                copernicus_password=copernicus_password,
                min_long=observed_bounds[0] if observed_bounds else None,
                max_long=observed_bounds[1] if observed_bounds else None,
                min_lat=observed_bounds[2] if observed_bounds else None,
                max_lat=observed_bounds[3] if observed_bounds else None,
            )
            event_queue.put(
                {
                    "type": "env_download_done",
                    "environment": environment,
                }
            )
        except Exception:
            event_queue.put(
                {
                    "type": "env_download_error",
                    "environment": environment,
                    "message": traceback.format_exc(),
                }
            )

    def build_request(
        staged_zip: Path,
        run_id: str,
        observed_bounds: tuple[float, float, float, float],
        current_dataset_paths: list[str],
        wind_dataset_paths: list[str],
        environmental_offset_hours: float,
        environmental_offset_values: list[int] | None = None,
    ) -> ValidationRunRequest:
        run_mode = (run_mode_dropdown.value or "OPTIMIZATION").strip().upper()
        optimize_enabled = run_mode != "NO OPTIMIZATION"
        wind_drift_factor = None
        current_drift_factor = None
        if optimize_enabled:
            wdf_min = parse_float(wdf_min_field.value, "WDF min")
            wdf_max = parse_float(wdf_max_field.value, "WDF max")
            wdf_step = parse_float(wdf_step_field.value, "WDF step")

            cdf_min = parse_float(cdf_min_field.value, "CDF min")
            cdf_max = parse_float(cdf_max_field.value, "CDF max")
            cdf_step = parse_float(cdf_step_field.value, "CDF step")

        else:
            wind_drift_factor = parse_float(fixed_wdf_field.value, "Fixed WDF")
            current_drift_factor = parse_float(fixed_cdf_field.value, "Fixed CDF")
            # Ranges are not used when optimization is disabled.
            wdf_min = 0.0
            wdf_max = 0.05
            wdf_step = 0.0025
            cdf_min = 0.5
            cdf_max = 1.0
            cdf_step = 0.1

        start_index = int(start_index_field.value or 0)

        return ValidationRunRequest(
            config_name="main",
            environment=environment_dropdown.value or "2019",
            forcing_source=(forcing_source_dropdown.value or "COPERNICUS"),
            shp_zip=str(staged_zip),
            min_long=observed_bounds[0],
            max_long=observed_bounds[1],
            min_lat=observed_bounds[2],
            max_lat=observed_bounds[3],
            start_index=start_index,
            environmental_offset_hours=float(environmental_offset_hours),
            environmental_offset_values=environmental_offset_values,
            optimize_wdf_cdf=optimize_enabled,
            optimize_wdf_mode="fast",
            fast_particles_per_wdf=1,
            wdf_min=wdf_min,
            wdf_max=wdf_max,
            wdf_step=wdf_step,
            cdf_min=cdf_min,
            cdf_max=cdf_max,
            cdf_step=cdf_step,
            skip_animation=False,
            skip_simulation=False,
            skip_plots=False,
            wind_drift_factor=wind_drift_factor,
            current_drift_factor=current_drift_factor,
            run_name=run_id,
            current_dataset_path=(current_dataset_paths[0] if current_dataset_paths else None),
            wind_dataset_path=(wind_dataset_paths[0] if wind_dataset_paths else None),
            current_dataset_paths=current_dataset_paths,
            wind_dataset_paths=wind_dataset_paths,
            disable_environment_offset=bool(current_dataset_paths or wind_dataset_paths),
        )

    def worker_run_validation(requests: list[ValidationRunRequest]) -> None:
        controller = SimulationController()
        writer = QueueWriter(event_queue)

        def on_progress(done: int, total: int) -> None:
            event_queue.put({"type": "progress", "done": done, "total": total})

        try:
            last_result = None
            total_runs = len(requests)
            for index, request in enumerate(requests, start=1):
                if state["cancel_event"].is_set():
                    event_queue.put({"type": "cancelled"})
                    return
                event_queue.put(
                    {
                        "type": "batch_start",
                        "index": index,
                        "total": total_runs,
                        "run_id": request.run_name,
                        "offset": request.environmental_offset_hours,
                        "offset_values": request.environmental_offset_values,
                    }
                )
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    result = controller.run_validation(
                        request,
                        progress_callback=on_progress,
                        should_cancel=state["cancel_event"].is_set,
                        show_plots=False,
                    )
                writer.flush()
                if not result:
                    event_queue.put({"type": "error", "message": "No result returned from run_validation."})
                    return
                last_result = result
                event_queue.put(
                    {
                        "type": "batch_done",
                        "index": index,
                        "total": total_runs,
                        "result": result,
                        "offset": request.environmental_offset_hours,
                        "offset_values": request.environmental_offset_values,
                    }
                )

            if last_result:
                if state["cancel_event"].is_set():
                    event_queue.put(
                        {
                            "type": "log",
                            "message": "Cancel requested late, but outputs are complete. Finalizing as done.",
                        }
                    )
                event_queue.put({"type": "done", "result": last_result})
                return
            if state["cancel_event"].is_set():
                event_queue.put({"type": "cancelled"})
                return
            event_queue.put({"type": "error", "message": "No validation requests were executed."})
        except RuntimeError as exc:
            writer.flush()
            if "cancel" in str(exc).lower():
                event_queue.put({"type": "cancelled"})
            else:
                event_queue.put({"type": "error", "message": traceback.format_exc()})
        except Exception:
            writer.flush()
            event_queue.put({"type": "error", "message": traceback.format_exc()})

    def build_stochastic_request(
        staged_zip: Path,
        run_id: str,
        observed_bounds: tuple[float, float, float, float],
        current_dataset_paths: list[str],
        wind_dataset_paths: list[str],
    ) -> StochasticValidationRunRequest:
        seed_raw = (stochastic_seed_field.value or "").strip()
        seed = int(seed_raw) if seed_raw else None
        n_simulations = int(stochastic_n_simulations_field.value or 0)
        number_of_workers = int(stochastic_number_of_workers_field.value or 4)
        run_name = (stochastic_run_name_field.value or "").strip() or run_id
        margin = (
            parse_float(stochastic_grid_margin_field.value, "Grid margin")
            if (stochastic_grid_margin_field.value or "").strip()
            else STOCHASTIC_GRID_MARGIN_DEG
        )
        grid = StochasticGridConfig(
            lon_min=_parse_optional_float(stochastic_grid_lon_min_field.value, "Lon min")
            if stochastic_grid_lon_min_field.value
            else observed_bounds[0],
            lon_max=_parse_optional_float(stochastic_grid_lon_max_field.value, "Lon max")
            if stochastic_grid_lon_max_field.value
            else observed_bounds[1],
            lat_min=_parse_optional_float(stochastic_grid_lat_min_field.value, "Lat min")
            if stochastic_grid_lat_min_field.value
            else observed_bounds[2],
            lat_max=_parse_optional_float(stochastic_grid_lat_max_field.value, "Lat max")
            if stochastic_grid_lat_max_field.value
            else observed_bounds[3],
            spatial_resolution=parse_float(stochastic_grid_resolution_field.value, "Grid resolution"),
            margin=margin,
        )
        stochastic_config = StochasticRunConfig(
            run_name=run_name,
            n_simulations=n_simulations,
            seed=seed,
            number_of_workers=number_of_workers,
            output_root=project_root / "data" / "2-simulated" / "stochastic",
            grid=grid,
            cdf=StochasticParameterConfig(
                enabled=bool(stochastic_cdf_enabled_checkbox.value),
                mean=parse_float(stochastic_cdf_mean_field.value, "CDF mean"),
                std=parse_float(stochastic_cdf_std_field.value, "CDF std"),
                min_value=parse_float(stochastic_cdf_min_field.value, "CDF min"),
                max_value=parse_float(stochastic_cdf_max_field.value, "CDF max"),
                default_value=parse_float(stochastic_cdf_mean_field.value, "CDF mean"),
            ),
            wdf=StochasticParameterConfig(
                enabled=bool(stochastic_wdf_enabled_checkbox.value),
                mean=parse_float(stochastic_wdf_mean_field.value, "WDF mean"),
                std=parse_float(stochastic_wdf_std_field.value, "WDF std"),
                min_value=parse_float(stochastic_wdf_min_field.value, "WDF min"),
                max_value=parse_float(stochastic_wdf_max_field.value, "WDF max"),
                default_value=parse_float(stochastic_wdf_mean_field.value, "WDF mean"),
            ),
            temporal_lag=TemporalLagConfig(
                enabled=bool(stochastic_tau_enabled_checkbox.value),
                mean=parse_float(stochastic_tau_mean_field.value, "Tau mean"),
                std=parse_float(stochastic_tau_std_field.value, "Tau std"),
                min_value=parse_float(stochastic_tau_min_field.value, "Tau min"),
                max_value=parse_float(stochastic_tau_max_field.value, "Tau max"),
                input_unit=stochastic_tau_input_unit_dropdown.value or "minutes",
                rounding_granularity=stochastic_tau_rounding_dropdown.value or "minutes",
                default_seconds=0.0,
            ),
        )

        return StochasticValidationRunRequest(
            stochastic_config=stochastic_config,
            config_name="main",
            environment=stochastic_environment_dropdown.value or "2019",
            forcing_source=stochastic_forcing_source_dropdown.value or "COPERNICUS",
            shp_zip=str(staged_zip),
            min_long=observed_bounds[0],
            max_long=observed_bounds[1],
            min_lat=observed_bounds[2],
            max_lat=observed_bounds[3],
            start_index=int(stochastic_start_index_field.value or 1),
            padding_animation_frame=0.1,
            run_name=run_name,
            current_dataset_path=(current_dataset_paths[0] if current_dataset_paths else None),
            wind_dataset_path=(wind_dataset_paths[0] if wind_dataset_paths else None),
            current_dataset_paths=current_dataset_paths,
            wind_dataset_paths=wind_dataset_paths,
            environmental_offset_hours=parse_float(
                stochastic_base_environmental_offset_hours_field.value,
                "Base env offset (h)",
            ),
        )

    def worker_run_stochastic(request: StochasticValidationRunRequest) -> None:
        controller = SimulationController()
        writer = QueueWriter(event_queue)

        def on_progress(done: int, total: int) -> None:
            event_queue.put({"type": "stochastic_progress", "done": done, "total": total})

        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                result = controller.run_stochastic_validation(
                    request,
                    progress_callback=on_progress,
                    should_cancel=state["cancel_event"].is_set,
                )
            writer.flush()
            event_queue.put({"type": "stochastic_done", "result": result})
        except RuntimeError as exc:
            writer.flush()
            if "cancel" in str(exc).lower():
                event_queue.put({"type": "cancelled"})
            else:
                event_queue.put({"type": "error", "message": traceback.format_exc()})
        except Exception:
            writer.flush()
            event_queue.put({"type": "error", "message": traceback.format_exc()})

    def start_execution(e: ft.ControlEvent) -> None:
        if state["running"]:
            status_title.value = "Blocked"
            status_subtitle.value = "Já existe execução em andamento."
            set_screen("Execution")
            append_log("Start blocked: there is already an execution running.")
            page.update()
            show_message("Já existe uma execução em andamento.", error=True)
            return
        if state["downloading"]:
            status_title.value = "Blocked"
            status_subtitle.value = "Download de ambiente em andamento."
            set_screen("Execution")
            append_log("Start blocked: wait for environment download to finish.")
            page.update()
            show_message(
                "Aguarde o término do download do ambiente para iniciar a execução.",
                error=True,
            )
            return

        set_screen("Execution")
        reset_execution_panel()
        status_title.value = "Preparing"
        status_subtitle.value = "Validando entradas..."
        append_log("Start requested by user.")
        page.update()

        try:
            if not state["selected_zip"]:
                raise ValueError("Selecione o ZIP com o spill observado.")
            source_zip = Path(state["selected_zip"])
            selected_current_datasets = [str(path) for path in state["selected_current_datasets"]]
            selected_wind_datasets = [str(path) for path in state["selected_wind_datasets"]]
            if selected_current_datasets:
                for current_dataset in selected_current_datasets:
                    current_path = Path(current_dataset)
                    if not current_path.exists():
                        raise ValueError(f"Arquivo de correnteza não encontrado: {current_path}")
            else:
                append_log("No custom current dataset selected; falling back to YAML water_dataset_path.")
            if selected_wind_datasets:
                for wind_dataset in selected_wind_datasets:
                    wind_path = Path(wind_dataset)
                    if not wind_path.exists():
                        raise ValueError(f"Arquivo de vento não encontrado: {wind_path}")
            else:
                append_log("No custom wind dataset selected; wind forcing depends on available readers.")
            has_prj = validate_observed_zip(source_zip)
            observed_bounds_raw = extract_observed_bounds(source_zip)
            observed_bounds = _expand_bounds(observed_bounds_raw)
            offset_values = build_offset_values()
            if not has_prj:
                append_log("Warning: observed ZIP has no .prj file.")
            base_run_id = build_run_id()
            requests = []
            run_mode = (run_mode_dropdown.value or "OPTIMIZATION").strip().upper()
            optimize_enabled = run_mode != "NO OPTIMIZATION"
            if optimize_enabled:
                run_id = f"{base_run_id}_envopt"
                staged_zip = stage_observed_zip(project_root, run_id, source_zip)
                requests.append(
                    build_request(
                        staged_zip=staged_zip,
                        run_id=run_id,
                        observed_bounds=observed_bounds,
                        current_dataset_paths=selected_current_datasets,
                        wind_dataset_paths=selected_wind_datasets,
                        environmental_offset_hours=offset_values[0],
                        environmental_offset_values=offset_values,
                    )
                )
            else:
                for offset_hours in offset_values:
                    run_id = f"{base_run_id}_{offset_suffix(offset_hours)}"
                    staged_zip = stage_observed_zip(project_root, run_id, source_zip)
                    requests.append(
                        build_request(
                            staged_zip=staged_zip,
                            run_id=run_id,
                            observed_bounds=observed_bounds,
                            current_dataset_paths=selected_current_datasets,
                            wind_dataset_paths=selected_wind_datasets,
                            environmental_offset_hours=offset_hours,
                        )
                    )
            state["run_id"] = requests[0].run_name if requests else base_run_id
            state["staged_zip"] = requests[0].shp_zip if requests else None
            append_log(
                "Observed bounds (raw) "
                f"lon=[{observed_bounds_raw[0]:.5f},{observed_bounds_raw[1]:.5f}] "
                f"lat=[{observed_bounds_raw[2]:.5f},{observed_bounds_raw[3]:.5f}]"
            )
            append_log(
                f"Observed bounds (padded {OBSERVED_BOUNDS_PADDING_DEG:.2f} deg, "
                f"min-span {MIN_OBSERVED_BOUNDS_SPAN_DEG:.2f} deg) "
                f"lon=[{observed_bounds[0]:.5f},{observed_bounds[1]:.5f}] "
                f"lat=[{observed_bounds[2]:.5f},{observed_bounds[3]:.5f}]"
            )
            append_log(
                (
                    "Environmental offsets to optimize: "
                    if optimize_enabled
                    else "Environmental offsets to run: "
                )
                + ", ".join(str(offset) for offset in offset_values)
            )
        except Exception as exc:
            status_title.value = "Failed"
            status_subtitle.value = "Falha na validação de entrada."
            append_log(traceback.format_exc())
            page.update()
            show_message(str(exc), error=True)
            return

        state["running"] = True
        state["cancel_event"].clear()
        state["start_time"] = time.monotonic()
        reset_execution_panel()
        append_log(f"First Run ID: {state['run_id']}")
        append_log(f"First staged observed ZIP: {state['staged_zip']}")
        page.update()

        thread = threading.Thread(target=worker_run_validation, args=(requests,), daemon=True)
        thread.start()

    def start_stochastic_execution(e: ft.ControlEvent) -> None:
        if state["running"]:
            status_title.value = "Blocked"
            status_subtitle.value = "Já existe execução em andamento."
            set_screen("Execution")
            append_log("Stochastic start blocked: there is already an execution running.")
            page.update()
            show_message("Já existe uma execução em andamento.", error=True)
            return
        if state["downloading"]:
            show_message("Aguarde o término do download do ambiente.", error=True)
            return

        set_screen("Execution")
        reset_execution_panel()
        status_title.value = "Preparing"
        status_subtitle.value = "Validando configuração estocástica..."
        append_log("Stochastic simulation requested by user.")
        page.update()

        try:
            if not state["selected_zip"]:
                raise ValueError("Selecione o ZIP com o spill observado.")
            source_zip = Path(state["selected_zip"])
            selected_current_datasets = [str(path) for path in state["selected_current_datasets"]]
            selected_wind_datasets = [str(path) for path in state["selected_wind_datasets"]]
            for current_dataset in selected_current_datasets:
                current_path = Path(current_dataset)
                if not current_path.exists():
                    raise ValueError(f"Arquivo de correnteza não encontrado: {current_path}")
            for wind_dataset in selected_wind_datasets:
                wind_path = Path(wind_dataset)
                if not wind_path.exists():
                    raise ValueError(f"Arquivo de vento não encontrado: {wind_path}")

            has_prj = validate_observed_zip(source_zip)
            observed_bounds_raw = extract_observed_bounds(source_zip)
            observed_bounds = _expand_bounds(observed_bounds_raw)
            run_id = build_run_id().replace("validation", "stochastic", 1)
            staged_zip = stage_observed_zip(project_root, run_id, source_zip)
            request = build_stochastic_request(
                staged_zip=staged_zip,
                run_id=run_id,
                observed_bounds=observed_bounds,
                current_dataset_paths=selected_current_datasets,
                wind_dataset_paths=selected_wind_datasets,
            )
            if not has_prj:
                append_log("Warning: observed ZIP has no .prj file.")
            state["run_id"] = request.stochastic_config.run_name
            state["staged_zip"] = str(staged_zip)
            append_log(
                "Observed bounds (raw) "
                f"lon=[{observed_bounds_raw[0]:.5f},{observed_bounds_raw[1]:.5f}] "
                f"lat=[{observed_bounds_raw[2]:.5f},{observed_bounds_raw[3]:.5f}]"
            )
            append_log(
                "Stochastic grid bounds "
                f"lon=[{request.stochastic_config.grid.lon_min:.5f},"
                f"{request.stochastic_config.grid.lon_max:.5f}] "
                f"lat=[{request.stochastic_config.grid.lat_min:.5f},"
                f"{request.stochastic_config.grid.lat_max:.5f}] "
                f"resolution={request.stochastic_config.grid.spatial_resolution:g}"
            )
        except Exception as exc:
            status_title.value = "Failed"
            status_subtitle.value = "Falha na validação de entrada."
            append_log(traceback.format_exc())
            page.update()
            show_message(str(exc), error=True)
            return

        state["running"] = True
        state["cancel_event"].clear()
        state["start_time"] = time.monotonic()
        reset_execution_panel()
        append_log(f"Run ID: {state['run_id']}")
        append_log(f"Staged observed ZIP: {state['staged_zip']}")
        page.update()

        thread = threading.Thread(target=worker_run_stochastic, args=(request,), daemon=True)
        thread.start()

    def start_environment_download(e: ft.ControlEvent) -> None:
        if state["running"]:
            show_message("Não é possível baixar dados durante uma execução.", error=True)
            return
        if state["downloading"]:
            show_message("Já existe um download em andamento.", error=True)
            return

        environment = environment_dropdown.value or "2019"
        copernicus_username = (copernicus_username_field.value or "").strip()
        copernicus_password = copernicus_password_field.value or ""
        if not copernicus_username or not copernicus_password:
            show_message(
                "Informe usuário e senha do Copernicus para baixar os dados.",
                error=True,
            )
            return
        observed_bounds = None
        force_download = False
        if state["selected_zip"]:
            try:
                source_zip = Path(state["selected_zip"])
                validate_observed_zip(source_zip)
                observed_bounds = _expand_bounds(extract_observed_bounds(source_zip))
                force_download = True
            except Exception as exc:
                show_message(f"Falha ao ler bounds do ZIP observado: {exc}", error=True)
                return
        state["downloading"] = True
        environment_download_status_text.value = f"{environment}: baixando dados..."
        status_title.value = "Downloading"
        status_subtitle.value = f"Baixando dados do ambiente {environment}..."
        if observed_bounds:
            append_log(
                f"Environment {environment}: starting Copernicus sal_temp download with observed bounds "
                f"lon=[{observed_bounds[0]:.5f},{observed_bounds[1]:.5f}] "
                f"lat=[{observed_bounds[2]:.5f},{observed_bounds[3]:.5f}] (force overwrite)."
            )
        else:
            append_log(f"Environment {environment}: starting Copernicus sal_temp download.")
        set_screen("Execution")
        page.update()

        thread = threading.Thread(
            target=worker_download_environment,
            args=(
                environment,
                copernicus_username,
                copernicus_password,
                observed_bounds,
                force_download,
            ),
            daemon=True,
        )
        thread.start()

    def cancel_execution(e: ft.ControlEvent) -> None:
        if not state["running"]:
            return
        state["cancel_event"].set()
        status_title.value = "Cancelling"
        status_subtitle.value = "Aguardando ponto seguro para parar..."
        append_log("Cancel requested by user.")
        page.update()

    # Selecao de arquivos. No Flet classico (0.28) pick_files() NAO e awaitable:
    # dispara o dialogo e o resultado chega pelo callback on_result. Como ha
    # tres botoes diferentes, guardamos qual handler deve tratar o proximo
    # resultado e roteamos em _on_file_picker_result.
    _pending_file_handler: dict = {"fn": None}

    def _warn_no_path() -> None:
        show_message(
            "Nao foi possivel obter o caminho do arquivo. No modo web o navegador "
            "nao expoe o caminho local do arquivo - rode o app em modo desktop.",
            error=True,
        )

    def _result_choose_zip(files: list) -> None:
        if not files:
            return
        selected_path = files[0].path
        if not selected_path:
            _warn_no_path()
            return
        state["selected_zip"] = selected_path
        selected_zip_text.value = selected_path
        stochastic_selected_zip_text.value = selected_path
        page.update()

    def _result_choose_dataset(files: list, state_key: str, label_text, stochastic_label_text) -> None:
        if not files:
            return
        selected_paths = [item.path for item in files if item.path]
        if not selected_paths:
            _warn_no_path()
            return
        state[state_key] = selected_paths
        selected_label = (
            f"{len(selected_paths)} arquivo(s): "
            + ", ".join(Path(path).name for path in selected_paths[:3])
        )
        if len(selected_paths) > 3:
            selected_label += ", ..."
        label_text.value = selected_label
        stochastic_label_text.value = selected_label
        page.update()

    def _on_file_picker_result(e: "ft.FilePickerResultEvent") -> None:
        handler = _pending_file_handler["fn"]
        _pending_file_handler["fn"] = None
        if handler is not None:
            handler(e.files or [])

    _file_type_custom = getattr(ft, "FilePickerFileType", None)
    _file_type_custom = _file_type_custom.CUSTOM if _file_type_custom is not None else "custom"

    def choose_zip(e: ft.ControlEvent) -> None:
        _pending_file_handler["fn"] = _result_choose_zip
        file_picker.pick_files(
            allow_multiple=False,
            file_type=_file_type_custom,
            allowed_extensions=["zip"],
        )

    def choose_current_dataset(e: ft.ControlEvent) -> None:
        _pending_file_handler["fn"] = lambda files: _result_choose_dataset(
            files, "selected_current_datasets",
            selected_current_dataset_text, stochastic_selected_current_dataset_text,
        )
        file_picker.pick_files(
            allow_multiple=True,
            file_type=_file_type_custom,
            allowed_extensions=["nc"],
        )

    def choose_wind_dataset(e: ft.ControlEvent) -> None:
        _pending_file_handler["fn"] = lambda files: _result_choose_dataset(
            files, "selected_wind_datasets",
            selected_wind_dataset_text, stochastic_selected_wind_dataset_text,
        )
        file_picker.pick_files(
            allow_multiple=True,
            file_type=_file_type_custom,
            allowed_extensions=["nc"],
        )

    file_picker = ft.FilePicker(on_result=_on_file_picker_result)
    # Flet classico (<=0.28) registra o FilePicker em page.overlay; versoes
    # novas (1.x) usam page.services. Suporta ambos.
    if hasattr(page, "overlay"):
        page.overlay.append(file_picker)
    else:
        page.services.append(file_picker)

    views = {
        "Validation Setup": build_setup_view(
            SetupViewBindings(
                choose_zip=choose_zip,
                selected_zip_text=selected_zip_text,
                choose_current_dataset=choose_current_dataset,
                selected_current_dataset_text=selected_current_dataset_text,
                choose_wind_dataset=choose_wind_dataset,
                selected_wind_dataset_text=selected_wind_dataset_text,
                forcing_source_dropdown=forcing_source_dropdown,
                environment_dropdown=environment_dropdown,
                copernicus_username_field=copernicus_username_field,
                copernicus_password_field=copernicus_password_field,
                environment_download_status_text=environment_download_status_text,
                download_environment_data=start_environment_download,
                start_index_field=start_index_field,
                environmental_offset_hours_field=environmental_offset_hours_field,
                run_mode_dropdown=run_mode_dropdown,
                fixed_wdf_field=fixed_wdf_field,
                fixed_cdf_field=fixed_cdf_field,
                wdf_min_field=wdf_min_field,
                wdf_max_field=wdf_max_field,
                wdf_step_field=wdf_step_field,
                cdf_min_field=cdf_min_field,
                cdf_max_field=cdf_max_field,
                cdf_step_field=cdf_step_field,
                start_execution=start_execution,
            )
        ),
        "Simulação Estocástica": build_stochastic_view(
            StochasticViewBindings(
                choose_zip=choose_zip,
                selected_zip_text=stochastic_selected_zip_text,
                choose_current_dataset=choose_current_dataset,
                selected_current_dataset_text=stochastic_selected_current_dataset_text,
                choose_wind_dataset=choose_wind_dataset,
                selected_wind_dataset_text=stochastic_selected_wind_dataset_text,
                forcing_source_dropdown=stochastic_forcing_source_dropdown,
                environment_dropdown=stochastic_environment_dropdown,
                start_index_field=stochastic_start_index_field,
                base_environmental_offset_hours_field=stochastic_base_environmental_offset_hours_field,
                run_name_field=stochastic_run_name_field,
                n_simulations_field=stochastic_n_simulations_field,
                number_of_workers_field=stochastic_number_of_workers_field,
                seed_field=stochastic_seed_field,
                cdf_enabled_checkbox=stochastic_cdf_enabled_checkbox,
                cdf_mean_field=stochastic_cdf_mean_field,
                cdf_std_field=stochastic_cdf_std_field,
                cdf_min_field=stochastic_cdf_min_field,
                cdf_max_field=stochastic_cdf_max_field,
                wdf_enabled_checkbox=stochastic_wdf_enabled_checkbox,
                wdf_mean_field=stochastic_wdf_mean_field,
                wdf_std_field=stochastic_wdf_std_field,
                wdf_min_field=stochastic_wdf_min_field,
                wdf_max_field=stochastic_wdf_max_field,
                tau_enabled_checkbox=stochastic_tau_enabled_checkbox,
                tau_mean_field=stochastic_tau_mean_field,
                tau_std_field=stochastic_tau_std_field,
                tau_min_field=stochastic_tau_min_field,
                tau_max_field=stochastic_tau_max_field,
                tau_input_unit_dropdown=stochastic_tau_input_unit_dropdown,
                tau_rounding_dropdown=stochastic_tau_rounding_dropdown,
                grid_lon_min_field=stochastic_grid_lon_min_field,
                grid_lon_max_field=stochastic_grid_lon_max_field,
                grid_lat_min_field=stochastic_grid_lat_min_field,
                grid_lat_max_field=stochastic_grid_lat_max_field,
                grid_resolution_field=stochastic_grid_resolution_field,
                grid_margin_field=stochastic_grid_margin_field,
                start_stochastic_execution=start_stochastic_execution,
            )
        ),
        "Execution": build_execution_view(
            ExecutionViewBindings(
                status_title=status_title,
                status_subtitle=status_subtitle,
                progress_text=progress_text,
                progress_bar=progress_bar,
                log_view=log_view,
                cancel_execution=cancel_execution,
            )
        ),
        "Results": build_results_view(
            ResultsViewBindings(
                result_score_text=result_score_text,
                result_runtime_text=result_runtime_text,
                best_wdf_text=best_wdf_text,
                best_cdf_text=best_cdf_text,
                best_environmental_offset_text=best_environmental_offset_text,
                frame_image=frame_image,
                frame_label=frame_label,
                frame_slider=frame_slider,
            )
        ),
        "Artifacts": build_artifacts_view(
            ArtifactsViewBindings(
                output_path_text=output_path_text,
                artifacts_column=artifacts_column,
                open_output_folder=lambda _: open_path(state["out_dir"]) if state["out_dir"] else None,
            )
        ),
    }

    content_host = ft.Container(expand=True)
    nav_buttons: dict[str, ft.Control] = {}

    def set_screen(screen_name: str) -> None:
        content_host.content = views[screen_name]
        for label, button in nav_buttons.items():
            button.style = ft.ButtonStyle(
                bgcolor="#364B6A" if label == screen_name else "transparent",
                color="#FFFFFF" if label == screen_name else "#D2D9E6",
                shape=ft.RoundedRectangleBorder(radius=10),
            )
        page.update()

    sidebar, created_nav_buttons = build_sidebar(on_navigate=set_screen)
    nav_buttons.update(created_nav_buttons)

    page.add(
        ft.Row(
            expand=True,
            spacing=0,
            controls=[
                sidebar,
                ft.Container(expand=True, content=content_host),
            ],
        )
    )

    set_screen("Validation Setup")


if __name__ == "__main__":
    import os

    # Por padrao abre a janela nativa (desktop). Se o cliente desktop do Flet
    # nao puder subir (ex.: falta libmpv.so.1 no sistema), rode em modo web:
    #   FLET_VIEW=web python -m src.adapters.flet_app
    # e abra http://localhost:8550 no navegador.
    if os.environ.get("FLET_VIEW", "desktop").lower() == "web":
        ft.app(target=main, view=ft.AppView.WEB_BROWSER, port=8550)
    else:
        ft.app(target=main)
