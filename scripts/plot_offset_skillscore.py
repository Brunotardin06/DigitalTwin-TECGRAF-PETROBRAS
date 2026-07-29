from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OFFSET_SUFFIX_RE = re.compile(r"_env(?P<sign>[mp])(?P<hours>\d+)$")


def parse_offset_from_name(path: Path) -> int | None:
    match = OFFSET_SUFFIX_RE.search(path.name)
    if not match:
        return None
    hours = int(match.group("hours"))
    return -hours if match.group("sign") == "m" else hours


def read_offset(run_dir: Path) -> int | None:
    for json_path in sorted(run_dir.glob("sim_*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "environmental_offset_hours" in payload:
            return int(round(float(payload["environmental_offset_hours"])))
    return parse_offset_from_name(run_dir)


def read_skillscore(run_dir: Path) -> float | None:
    summary_path = run_dir / "wdf_cdf_optimization_fast.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("skillscore") is not None:
            return float(payload["skillscore"])

    csv_path = run_dir / "wdf_cdf_optimization_fast.csv"
    if not csv_path.exists():
        return None
    dataframe = pd.read_csv(csv_path)
    if dataframe.empty or "skillscore" not in dataframe.columns:
        return None
    valid = dataframe[dataframe["skillscore"].notna()]
    if valid.empty:
        return None
    return float(valid["skillscore"].max())


def collect_scores(root: Path, min_offset: int, max_offset: int) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        offset = read_offset(run_dir)
        if offset is None or offset < min_offset or offset > max_offset:
            continue
        skillscore = read_skillscore(run_dir)
        if skillscore is None:
            continue
        rows.append(
            {
                "offset_hours": offset,
                "skillscore": skillscore,
                "run_dir": str(run_dir),
            }
        )

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        return dataframe
    return (
        dataframe.sort_values(["offset_hours", "skillscore"], ascending=[True, False])
        .drop_duplicates("offset_hours", keep="first")
        .sort_values("offset_hours")
        .reset_index(drop=True)
    )


def plot_scores(dataframe: pd.DataFrame, output_path: Path, min_offset: int, max_offset: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        dataframe["offset_hours"],
        dataframe["skillscore"],
        marker="o",
        linewidth=2,
        color="#1f77b4",
    )
    ax.axvline(0, color="#444444", linewidth=1, linestyle="--")
    ax.set_xlim(min_offset, max_offset)
    ax.set_xticks(list(range(min_offset, max_offset + 1)))
    ax.set_xlabel("Defasagem ambiental (h)")
    ax.set_ylabel("Skillscore")
    ax.set_title("Variação de desempenho por defasagem ambiental")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera grafico skillscore x defasagem ambiental usando resultados do lote."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/2-simulated"),
        help="Diretorio que contem as pastas validation_*_envm/envp.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/2-simulated/offset_skillscore_-20_5.png"),
        help="Arquivo PNG de saida.",
    )
    parser.add_argument("--min-offset", type=int, default=-20)
    parser.add_argument("--max-offset", type=int, default=5)
    args = parser.parse_args()

    dataframe = collect_scores(args.root, args.min_offset, args.max_offset)
    if dataframe.empty:
        raise SystemExit(
            f"Nenhum resultado encontrado em {args.root} para offsets "
            f"{args.min_offset}..{args.max_offset}."
        )

    csv_path = args.out.with_suffix(".csv")
    dataframe.to_csv(csv_path, index=False)
    plot_scores(dataframe, args.out, args.min_offset, args.max_offset)

    print(f"Saved plot: {args.out}")
    print(f"Saved data: {csv_path}")
    print(dataframe[["offset_hours", "skillscore"]].to_string(index=False))


if __name__ == "__main__":
    main()
