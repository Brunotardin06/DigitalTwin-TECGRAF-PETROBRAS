from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import flet as ft


def _icon_value(name: str, fallback: str):
    icons_obj = getattr(ft, "Icons", None)
    if icons_obj is None:
        icons_obj = getattr(ft, "icons", None)
    if icons_obj is not None:
        return getattr(icons_obj, name, fallback)
    return fallback


def _enum_value(enum_name: str, member_name: str, fallback):
    enum_obj = getattr(ft, enum_name, None)
    if enum_obj is None:
        return fallback
    return getattr(enum_obj, member_name, fallback)


@dataclass(frozen=True)
class SetupViewBindings:
    choose_zip: Callable[[ft.ControlEvent], None]
    selected_zip_text: ft.Text
    choose_current_dataset: Callable[[ft.ControlEvent], None]
    selected_current_dataset_text: ft.Text
    choose_wind_dataset: Callable[[ft.ControlEvent], None]
    selected_wind_dataset_text: ft.Text
    forcing_source_dropdown: ft.Dropdown
    environment_dropdown: ft.Dropdown
    copernicus_username_field: ft.TextField
    copernicus_password_field: ft.TextField
    environment_download_status_text: ft.Text
    download_environment_data: Callable[[ft.ControlEvent], None]
    start_index_field: ft.TextField
    environmental_offset_hours_field: ft.TextField
    run_mode_dropdown: ft.Dropdown
    fixed_wdf_field: ft.TextField
    fixed_cdf_field: ft.TextField
    wdf_min_field: ft.TextField
    wdf_max_field: ft.TextField
    wdf_step_field: ft.TextField
    cdf_min_field: ft.TextField
    cdf_max_field: ft.TextField
    cdf_step_field: ft.TextField
    start_execution: Callable[[ft.ControlEvent], None]


@dataclass(frozen=True)
class StochasticViewBindings:
    choose_zip: Callable[[ft.ControlEvent], None]
    selected_zip_text: ft.Text
    choose_current_dataset: Callable[[ft.ControlEvent], None]
    selected_current_dataset_text: ft.Text
    choose_wind_dataset: Callable[[ft.ControlEvent], None]
    selected_wind_dataset_text: ft.Text
    forcing_source_dropdown: ft.Dropdown
    environment_dropdown: ft.Dropdown
    copernicus_username_field: ft.TextField
    copernicus_password_field: ft.TextField
    environment_download_status_text: ft.Text
    download_environment_data: Callable[[ft.ControlEvent], None]
    start_index_field: ft.TextField
    base_environmental_offset_hours_field: ft.TextField
    run_name_field: ft.TextField
    n_simulations_field: ft.TextField
    number_of_workers_field: ft.TextField
    cdf_enabled_checkbox: ft.Checkbox
    cdf_mean_field: ft.TextField
    cdf_std_field: ft.TextField
    cdf_min_field: ft.TextField
    cdf_max_field: ft.TextField
    wdf_enabled_checkbox: ft.Checkbox
    wdf_mean_field: ft.TextField
    wdf_std_field: ft.TextField
    wdf_min_field: ft.TextField
    wdf_max_field: ft.TextField
    tau_enabled_checkbox: ft.Checkbox
    tau_mean_field: ft.TextField
    tau_std_field: ft.TextField
    tau_min_field: ft.TextField
    tau_max_field: ft.TextField
    tau_input_unit_dropdown: ft.Dropdown
    tau_rounding_dropdown: ft.Dropdown
    grid_lon_min_field: ft.TextField
    grid_lon_max_field: ft.TextField
    grid_lat_min_field: ft.TextField
    grid_lat_max_field: ft.TextField
    grid_resolution_field: ft.TextField
    grid_margin_field: ft.TextField
    start_stochastic_execution: Callable[[ft.ControlEvent], None]


@dataclass(frozen=True)
class DeterministicViewBindings:
    lon_field: ft.TextField
    lat_field: ft.TextField
    date_field: ft.TextField
    time_field: ft.TextField
    duration_days_field: ft.TextField
    leak_duration_hours_field: ft.TextField
    seed_radius_field: ft.TextField
    num_seed_elements_field: ft.TextField
    time_step_field: ft.TextField
    output_time_step_field: ft.TextField
    wdf_field: ft.TextField
    cdf_field: ft.TextField
    oil_type_field: ft.TextField
    forcing_source_dropdown: ft.Dropdown
    environment_dropdown: ft.Dropdown
    use_sal_temp_checkbox: ft.Checkbox
    choose_current_dataset: Callable[[ft.ControlEvent], None]
    selected_current_dataset_text: ft.Text
    choose_wind_dataset: Callable[[ft.ControlEvent], None]
    selected_wind_dataset_text: ft.Text
    run_name_field: ft.TextField
    start_deterministic_execution: Callable[[ft.ControlEvent], None]


@dataclass(frozen=True)
class ExecutionViewBindings:
    status_title: ft.Text
    status_subtitle: ft.Text
    progress_text: ft.Text
    progress_bar: ft.ProgressBar
    log_view: ft.ListView
    cancel_execution: Callable[[ft.ControlEvent], None]


@dataclass(frozen=True)
class ResultsViewBindings:
    result_score_text: ft.Text
    result_runtime_text: ft.Text
    best_wdf_text: ft.Text
    best_cdf_text: ft.Text
    best_environmental_offset_text: ft.Text
    frame_image: ft.Image
    frame_label: ft.Text
    frame_slider: ft.Slider


@dataclass(frozen=True)
class ArtifactsViewBindings:
    output_path_text: ft.Text
    artifacts_column: ft.Column
    open_output_folder: Callable[[ft.ControlEvent], None]


def build_setup_view(bindings: SetupViewBindings) -> ft.Container:
    return ft.Container(
        padding=28,
        content=ft.Column(
            spacing=18,
            controls=[
                ft.Text("Simulação de Casos", size=42, weight=ft.FontWeight.W_700, color="#0F172A"),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Column(
                        spacing=14,
                        controls=[
                            ft.Text("Observed Spill Data", size=28, weight=ft.FontWeight.W_600),
                            ft.Row(
                                spacing=12,
                                controls=[
                                    ft.ElevatedButton(
                                        "Choose ZIP",
                                        icon=_icon_value("UPLOAD_FILE", "upload_file"),
                                        on_click=bindings.choose_zip,
                                    ),
                                    bindings.selected_zip_text,
                                ],
                            ),
                        ],
                    ),
                ),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Column(
                        spacing=14,
                        controls=[
                            ft.Text("Forcing Inputs (.nc)", size=28, weight=ft.FontWeight.W_600),
                            ft.Row(
                                spacing=12,
                                controls=[bindings.forcing_source_dropdown],
                            ),
                            ft.Row(
                                spacing=12,
                                controls=[
                                    ft.OutlinedButton(
                                        "Choose Current File(s)",
                                        icon=_icon_value("WATER_DROP", "water_drop"),
                                        on_click=bindings.choose_current_dataset,
                                    ),
                                    bindings.selected_current_dataset_text,
                                ],
                            ),
                            ft.Row(
                                spacing=12,
                                controls=[
                                    ft.OutlinedButton(
                                        "Choose Wind File(s)",
                                        icon=_icon_value("AIR", "air"),
                                        on_click=bindings.choose_wind_dataset,
                                    ),
                                    bindings.selected_wind_dataset_text,
                                ],
                            ),
                        ],
                    ),
                ),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Column(
                        spacing=14,
                        controls=[
                            ft.Text("Environment Configuration", size=28, weight=ft.FontWeight.W_600),
                            ft.Row(
                                spacing=16,
                                controls=[
                                    bindings.environment_dropdown,
                                    bindings.start_index_field,
                                    bindings.environmental_offset_hours_field,
                                ],
                            ),
                            ft.Row(
                                spacing=12,
                                controls=[
                                    bindings.copernicus_username_field,
                                    bindings.copernicus_password_field,
                                ],
                            ),
                            ft.Row(
                                spacing=12,
                                controls=[
                                    ft.OutlinedButton(
                                        "Download Environment Data",
                                        icon=_icon_value("DOWNLOAD", "download"),
                                        on_click=bindings.download_environment_data,
                                    ),
                                    bindings.environment_download_status_text,
                                ],
                            ),
                        ],
                    ),
                ),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Column(
                        spacing=14,
                        controls=[
                            ft.Text("Optimization Parameters", size=28, weight=ft.FontWeight.W_600),
                            bindings.run_mode_dropdown,
                            ft.Text("Fixed Parameters", size=20, weight=ft.FontWeight.W_600),
                            ft.Row(
                                spacing=12,
                                controls=[
                                    bindings.fixed_wdf_field,
                                    bindings.fixed_cdf_field,
                                ],
                            ),
                            ft.Text("Wind Drift Factor (WDF)", size=20, weight=ft.FontWeight.W_600),
                            ft.Row(
                                spacing=12,
                                controls=[
                                    bindings.wdf_min_field,
                                    bindings.wdf_max_field,
                                    bindings.wdf_step_field,
                                ],
                            ),
                            ft.Text("Current Drift Factor (CDF)", size=20, weight=ft.FontWeight.W_600),
                            ft.Row(
                                spacing=12,
                                controls=[
                                    bindings.cdf_min_field,
                                    bindings.cdf_max_field,
                                    bindings.cdf_step_field,
                                ],
                            ),
                        ],
                    ),
                ),
                ft.Row(
                    alignment=ft.MainAxisAlignment.END,
                    controls=[
                        ft.FilledButton(
                            "Start Execution",
                            icon=_icon_value("PLAY_ARROW", "play_arrow"),
                            on_click=bindings.start_execution,
                        )
                    ],
                ),
            ],
        ),
    )


def build_stochastic_view(bindings: StochasticViewBindings) -> ft.Container:
    return ft.Container(
        padding=28,
        content=ft.Column(
            spacing=18,
            scroll=_enum_value("ScrollMode", "AUTO", "auto"),
            controls=[
                ft.Text("Simulação Estocástica", size=42, weight=ft.FontWeight.W_700, color="#0F172A"),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Column(
                        spacing=14,
                        controls=[
                            ft.Text("Entradas", size=28, weight=ft.FontWeight.W_600),
                            ft.Row(
                                spacing=12,
                                controls=[
                                    ft.ElevatedButton(
                                        "Choose ZIP",
                                        icon=_icon_value("UPLOAD_FILE", "upload_file"),
                                        on_click=bindings.choose_zip,
                                    ),
                                    bindings.selected_zip_text,
                                ],
                            ),
                            ft.Row(
                                spacing=12,
                                controls=[bindings.forcing_source_dropdown, bindings.environment_dropdown],
                            ),
                            ft.Row(
                                spacing=12,
                                controls=[
                                    bindings.copernicus_username_field,
                                    bindings.copernicus_password_field,
                                ],
                            ),
                            ft.Row(
                                spacing=12,
                                controls=[
                                    ft.OutlinedButton(
                                        "Download Environment Data",
                                        icon=_icon_value("DOWNLOAD", "download"),
                                        on_click=bindings.download_environment_data,
                                    ),
                                    bindings.environment_download_status_text,
                                ],
                            ),
                            ft.Row(
                                spacing=12,
                                controls=[
                                    ft.OutlinedButton(
                                        "Choose Current File(s)",
                                        icon=_icon_value("WATER_DROP", "water_drop"),
                                        on_click=bindings.choose_current_dataset,
                                    ),
                                    bindings.selected_current_dataset_text,
                                ],
                            ),
                            ft.Row(
                                spacing=12,
                                controls=[
                                    ft.OutlinedButton(
                                        "Choose Wind File(s)",
                                        icon=_icon_value("AIR", "air"),
                                        on_click=bindings.choose_wind_dataset,
                                    ),
                                    bindings.selected_wind_dataset_text,
                                ],
                            ),
                        ],
                    ),
                ),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Column(
                        spacing=14,
                        controls=[
                            ft.Text("Execução", size=28, weight=ft.FontWeight.W_600),
                            ft.Row(
                                spacing=12,
                                controls=[
                                    bindings.run_name_field,
                                    bindings.n_simulations_field,
                                    bindings.number_of_workers_field,
                                    bindings.start_index_field,
                                    bindings.base_environmental_offset_hours_field,
                                ],
                            ),
                        ],
                    ),
                ),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Column(
                        spacing=14,
                        controls=[
                            ft.Text("CDF", size=28, weight=ft.FontWeight.W_600),
                            bindings.cdf_enabled_checkbox,
                            ft.Row(
                                spacing=12,
                                controls=[
                                    bindings.cdf_mean_field,
                                    bindings.cdf_std_field,
                                    bindings.cdf_min_field,
                                    bindings.cdf_max_field,
                                ],
                            ),
                        ],
                    ),
                ),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Column(
                        spacing=14,
                        controls=[
                            ft.Text("WDF", size=28, weight=ft.FontWeight.W_600),
                            bindings.wdf_enabled_checkbox,
                            ft.Row(
                                spacing=12,
                                controls=[
                                    bindings.wdf_mean_field,
                                    bindings.wdf_std_field,
                                    bindings.wdf_min_field,
                                    bindings.wdf_max_field,
                                ],
                            ),
                        ],
                    ),
                ),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Column(
                        spacing=14,
                        controls=[
                            ft.Text("Defasagem Temporal", size=28, weight=ft.FontWeight.W_600),
                            bindings.tau_enabled_checkbox,
                            ft.Row(
                                spacing=12,
                                controls=[
                                    bindings.tau_mean_field,
                                    bindings.tau_std_field,
                                    bindings.tau_min_field,
                                    bindings.tau_max_field,
                                    bindings.tau_input_unit_dropdown,
                                    bindings.tau_rounding_dropdown,
                                ],
                            ),
                        ],
                    ),
                ),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Column(
                        spacing=14,
                        controls=[
                            ft.Text("Grade Espacial Fixa", size=28, weight=ft.FontWeight.W_600),
                            ft.Row(
                                spacing=12,
                                controls=[
                                    bindings.grid_lon_min_field,
                                    bindings.grid_lon_max_field,
                                    bindings.grid_lat_min_field,
                                    bindings.grid_lat_max_field,
                                    bindings.grid_resolution_field,
                                    bindings.grid_margin_field,
                                ],
                            ),
                        ],
                    ),
                ),
                ft.Row(
                    alignment=ft.MainAxisAlignment.END,
                    controls=[
                        ft.FilledButton(
                            "Run Stochastic Simulation",
                            icon=_icon_value("PLAY_ARROW", "play_arrow"),
                            on_click=bindings.start_stochastic_execution,
                        )
                    ],
                ),
            ],
        ),
    )


def build_deterministic_view(bindings: DeterministicViewBindings) -> ft.Container:
    def _card(title: str, *rows: ft.Control) -> ft.Container:
        return ft.Container(
            bgcolor="#EDF1F6",
            border=ft.border.all(1, "#C8D1DB"),
            border_radius=12,
            padding=20,
            content=ft.Column(
                spacing=14,
                controls=[ft.Text(title, size=28, weight=ft.FontWeight.W_600), *rows],
            ),
        )

    return ft.Container(
        padding=28,
        content=ft.Column(
            spacing=18,
            scroll=_enum_value("ScrollMode", "AUTO", "auto"),
            controls=[
                ft.Text(
                    "Simulação Determinística",
                    size=42,
                    weight=ft.FontWeight.W_700,
                    color="#0F172A",
                ),
                ft.Text(
                    "Prevê a deriva de um vazamento a partir de um ponto, data e "
                    "duração, sem depender de manchas observadas.",
                    size=16,
                    color="#4B6385",
                ),
                _card(
                    "Local e Momento do Vazamento",
                    ft.Row(spacing=12, controls=[bindings.lon_field, bindings.lat_field]),
                    ft.Row(
                        spacing=12,
                        controls=[
                            bindings.date_field,
                            bindings.time_field,
                            bindings.duration_days_field,
                            bindings.leak_duration_hours_field,
                        ],
                    ),
                    ft.Row(
                        spacing=12,
                        controls=[
                            bindings.seed_radius_field,
                            bindings.num_seed_elements_field,
                        ],
                    ),
                ),
                _card(
                    "Forcing Inputs (.nc)",
                    ft.Row(
                        spacing=12,
                        controls=[
                            bindings.forcing_source_dropdown,
                            bindings.environment_dropdown,
                            bindings.use_sal_temp_checkbox,
                        ],
                    ),
                    ft.Row(
                        spacing=12,
                        controls=[
                            ft.OutlinedButton(
                                "Choose Current File(s)",
                                icon=_icon_value("WATER_DROP", "water_drop"),
                                on_click=bindings.choose_current_dataset,
                            ),
                            bindings.selected_current_dataset_text,
                        ],
                    ),
                    ft.Row(
                        spacing=12,
                        controls=[
                            ft.OutlinedButton(
                                "Choose Wind File(s)",
                                icon=_icon_value("AIR", "air"),
                                on_click=bindings.choose_wind_dataset,
                            ),
                            bindings.selected_wind_dataset_text,
                        ],
                    ),
                ),
                _card(
                    "Parâmetros do Modelo",
                    ft.Row(
                        spacing=12,
                        controls=[
                            bindings.wdf_field,
                            bindings.cdf_field,
                            bindings.oil_type_field,
                        ],
                    ),
                    ft.Row(
                        spacing=12,
                        controls=[
                            bindings.time_step_field,
                            bindings.output_time_step_field,
                            bindings.run_name_field,
                        ],
                    ),
                ),
                ft.Row(
                    alignment=ft.MainAxisAlignment.END,
                    controls=[
                        ft.FilledButton(
                            "Run Deterministic Simulation",
                            icon=_icon_value("PLAY_ARROW", "play_arrow"),
                            on_click=bindings.start_deterministic_execution,
                        )
                    ],
                ),
            ],
        ),
    )


def build_execution_view(bindings: ExecutionViewBindings) -> ft.Container:
    return ft.Container(
        padding=28,
        content=ft.Column(
            spacing=18,
            controls=[
                ft.Text("Execution Progress", size=42, weight=ft.FontWeight.W_700, color="#0F172A"),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Column(
                        spacing=16,
                        controls=[
                            ft.Row(
                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                controls=[
                                    ft.Column(
                                        spacing=4,
                                        controls=[bindings.status_title, bindings.status_subtitle],
                                    ),
                                    ft.OutlinedButton(
                                        "Cancel",
                                        icon=_icon_value("CANCEL", "cancel"),
                                        style=ft.ButtonStyle(color="#B91C1C"),
                                        on_click=bindings.cancel_execution,
                                    ),
                                ],
                            ),
                            bindings.progress_text,
                            bindings.progress_bar,
                        ],
                    ),
                ),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    expand=True,
                    content=ft.Column(
                        spacing=10,
                        expand=True,
                        controls=[
                            ft.Text("Execution Log", size=26, weight=ft.FontWeight.W_600),
                            bindings.log_view,
                        ],
                    ),
                ),
            ],
        ),
    )


def build_results_view(bindings: ResultsViewBindings) -> ft.Container:
    return ft.Container(
        padding=28,
        content=ft.Column(
            spacing=18,
            controls=[
                ft.Text("Results Overview", size=42, weight=ft.FontWeight.W_700, color="#0F172A"),
                ft.Row(
                    spacing=16,
                    controls=[
                        ft.Container(
                            width=270,
                            bgcolor="#EDF1F6",
                            border=ft.border.all(1, "#C8D1DB"),
                            border_radius=12,
                            padding=18,
                            content=ft.Column(
                                spacing=6,
                                controls=[
                                    ft.Text("Skill Score", size=18, color="#4B6385"),
                                    bindings.result_score_text,
                                ],
                            ),
                        ),
                        ft.Container(
                            width=270,
                            bgcolor="#EDF1F6",
                            border=ft.border.all(1, "#C8D1DB"),
                            border_radius=12,
                            padding=18,
                            content=ft.Column(
                                spacing=6,
                                controls=[
                                    ft.Text("Duration", size=18, color="#4B6385"),
                                    bindings.result_runtime_text,
                                ],
                            ),
                        ),
                    ],
                ),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Column(
                        spacing=10,
                        controls=[
                            ft.Text("Best Parameters", size=26, weight=ft.FontWeight.W_600),
                            ft.Row(
                                spacing=24,
                                controls=[
                                    ft.Column(controls=[ft.Text("WDF", color="#4B6385"), bindings.best_wdf_text]),
                                    ft.Column(controls=[ft.Text("CDF", color="#4B6385"), bindings.best_cdf_text]),
                                    ft.Column(
                                        controls=[
                                            ft.Text("Env offset", color="#4B6385"),
                                            bindings.best_environmental_offset_text,
                                        ]
                                    ),
                                ],
                            ),
                        ],
                    ),
                ),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Column(
                        spacing=12,
                        controls=[
                            ft.Text("Trajectory Comparison", size=26, weight=ft.FontWeight.W_600),
                            bindings.frame_image,
                            bindings.frame_label,
                            bindings.frame_slider,
                        ],
                    ),
                ),
            ],
        ),
    )


def build_artifacts_view(bindings: ArtifactsViewBindings) -> ft.Container:
    return ft.Container(
        padding=28,
        content=ft.Column(
            spacing=18,
            controls=[
                ft.Text("Artifacts & Export", size=42, weight=ft.FontWeight.W_700, color="#0F172A"),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Row(
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                        controls=[
                            ft.Column(
                                spacing=4,
                                controls=[
                                    ft.Text("Output Location", size=24, weight=ft.FontWeight.W_600),
                                    bindings.output_path_text,
                                ],
                            ),
                            ft.OutlinedButton(
                                "Open Folder",
                                icon=_icon_value("FOLDER_OPEN", "folder_open"),
                                on_click=bindings.open_output_folder,
                            ),
                        ],
                    ),
                ),
                ft.Container(
                    bgcolor="#EDF1F6",
                    border=ft.border.all(1, "#C8D1DB"),
                    border_radius=12,
                    padding=20,
                    content=ft.Column(
                        spacing=10,
                        controls=[
                            ft.Text("Generated Artifacts", size=26, weight=ft.FontWeight.W_600),
                            bindings.artifacts_column,
                        ],
                    ),
                ),
            ],
        ),
    )


def build_sidebar(on_navigate: Callable[[str], None]) -> tuple[ft.Container, dict[str, ft.TextButton]]:
    nav_buttons: dict[str, ft.TextButton] = {}
    nav_column = ft.Column(spacing=10)

    for label, icon in [
        ("Simulação de Casos", _icon_value("DESCRIPTION_OUTLINED", "description_outlined")),
        ("Simulação Determinística", _icon_value("PLACE_OUTLINED", "place_outlined")),
        ("Simulação Estocástica", _icon_value("SCIENCE_OUTLINED", "science_outlined")),
        ("Execution", _icon_value("PLAY_ARROW_OUTLINED", "play_arrow_outlined")),
        ("Results", _icon_value("BAR_CHART_OUTLINED", "bar_chart_outlined")),
        ("Artifacts", _icon_value("FOLDER_OPEN_OUTLINED", "folder_open_outlined")),
    ]:
        btn = ft.TextButton(
            text=label,
            icon=icon,
            style=ft.ButtonStyle(
                color="#D2D9E6",
                alignment=ft.Alignment(-1, 0),
                shape=ft.RoundedRectangleBorder(radius=10),
                padding=ft.padding.symmetric(horizontal=12, vertical=14),
            ),
            on_click=lambda e, name=label: on_navigate(name),
        )
        nav_buttons[label] = btn
        nav_column.controls.append(btn)

    sidebar = ft.Container(
        width=300,
        bgcolor="#1E2C45",
        padding=20,
        content=ft.Column(
            expand=True,
            spacing=20,
            controls=[
                ft.Text("Oil Spill Drift", size=40, weight=ft.FontWeight.W_700, color="#FFFFFF"),
                ft.Text(" ", size=18, color="#9DB0CE"),
                nav_column,
                ft.Container(expand=True),
            ],
        ),
    )
    return sidebar, nav_buttons
