from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


WINDOW_RE = re.compile(r"^(\d{1,2})_(\d{1,2})\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera tabela e graficos dos parametros otimizados (WDF, CDF, HD) "
            "a partir de JSONs por janela horaria de 4h."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("OTIMIZAÇÃO 12 DE JUNHO 4h"),
        help="Pasta com arquivos JSON no formato X_Y.json",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=3,
        help="Casas decimais para arredondar os parametros",
    )
    return parser.parse_args()


def load_rows(input_dir: Path, ndigits: int) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(input_dir.glob("*.json")):
        match = WINDOW_RE.match(path.name)
        if not match:
            continue
        start_h = int(match.group(1))
        end_h = int(match.group(2))
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "janela_inicio": start_h,
                "janela_fim": end_h,
                "janela": f"{start_h}-{end_h}",
                "wind_drift_factor": round(float(payload["wind_drift_factor"]), ndigits),
                "current_drift_factor": round(float(payload["current_drift_factor"]), ndigits),
                "horizontal_diffusivity": round(float(payload["horizontal_diffusivity"]), ndigits),
                "environment": payload.get("environment"),
                "simulation_name": payload.get("simulation_name"),
                "arquivo": path.name,
            }
        )

    if not rows:
        raise ValueError(f"Nenhum JSON valido encontrado em: {input_dir}")

    dataframe = pd.DataFrame(rows).sort_values(["janela_inicio", "janela_fim"]).reset_index(drop=True)
    return dataframe


def write_outputs(input_dir: Path, dataframe: pd.DataFrame) -> tuple[Path, Path, Path, Path]:
    csv_path = input_dir / "resumo_parametros_otimizados_4h.csv"
    fig_path = input_dir / "grafico_parametros_otimizados_4h.png"
    fig_all_path = input_dir / "grafico_parametros_otimizados_4h_cdf_wdf.png"
    fig_split_path = input_dir / "grafico_parametros_otimizados_4h_cdf_wdf_separado.png"

    dataframe.to_csv(csv_path, index=False, encoding="utf-8")

    labels = dataframe["janela"].tolist()
    x = range(len(labels))

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Evolucao dos parametros otimizados por janela horaria (4h)", fontsize=15)

    axes[0].plot(x, dataframe["wind_drift_factor"], marker="o", color="#1f77b4")
    axes[0].set_ylabel("WDF")
    axes[0].grid(alpha=0.25)

    axes[1].plot(x, dataframe["current_drift_factor"], marker="o", color="#ff7f0e")
    axes[1].set_ylabel("CDF")
    axes[1].grid(alpha=0.25)

    axes[2].plot(x, dataframe["horizontal_diffusivity"], marker="o", color="#2ca02c")
    axes[2].set_ylabel("HD")
    axes[2].set_xlabel("Janela horaria (h)")
    axes[2].grid(alpha=0.25)

    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(labels, rotation=45, ha="right")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    fig_all, ax_all = plt.subplots(figsize=(14, 5))
    ax_all.plot(x, dataframe["wind_drift_factor"], marker="o", color="#1f77b4", label="WDF")
    ax_all.plot(x, dataframe["current_drift_factor"], marker="o", color="#ff7f0e", label="CDF")
    ax_all.set_title("Parametros otimizados por janela horaria de 4h (CDF e WDF)")
    ax_all.set_xlabel("Janela horaria (h)")
    ax_all.set_ylabel("Valor do parametro")
    ax_all.grid(alpha=0.25)
    ax_all.set_xticks(list(x))
    ax_all.set_xticklabels(labels, rotation=45, ha="right")
    ax_all.legend()
    fig_all.tight_layout()
    fig_all.savefig(fig_all_path, dpi=160)
    plt.close(fig_all)

    fig_split, axes_split = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig_split.suptitle("Parametros otimizados por janela horaria de 4h (CDF e WDF separados)", fontsize=14)

    axes_split[0].plot(x, dataframe["wind_drift_factor"], marker="o", color="#1f77b4")
    axes_split[0].set_ylabel("WDF")
    axes_split[0].grid(alpha=0.25)

    axes_split[1].plot(x, dataframe["current_drift_factor"], marker="o", color="#ff7f0e")
    axes_split[1].set_ylabel("CDF")
    axes_split[1].set_xlabel("Janela horaria (h)")
    axes_split[1].grid(alpha=0.25)
    axes_split[1].set_xticks(list(x))
    axes_split[1].set_xticklabels(labels, rotation=45, ha="right")

    fig_split.tight_layout(rect=(0, 0, 1, 0.96))
    fig_split.savefig(fig_split_path, dpi=160)
    plt.close(fig_split)

    return csv_path, fig_path, fig_all_path, fig_split_path


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    dataframe = load_rows(input_dir=input_dir, ndigits=args.round_decimals)
    csv_path, fig_path, fig_all_path, fig_split_path = write_outputs(
        input_dir=input_dir,
        dataframe=dataframe,
    )
    print(f"OK: {csv_path}")
    print(f"OK: {fig_path}")
    print(f"OK: {fig_all_path}")
    print(f"OK: {fig_split_path}")


if __name__ == "__main__":
    main()
