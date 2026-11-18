"""Turn an nvidia-smi CSV log into a GPU-usage chart and a summary table for the report.

Record the log (in a separate tmux session) while training runs:
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
             --format=csv -l 1 | tee outputs/gpu_log.csv

Then plot it:
  python scripts/plot_gpu_log.py outputs/gpu_log.csv [--out outputs/gpu_usage.png] [--title "ConvNeXt-Tiny"]

Writes the chart (PNG) and <out>_summary.json, and prints mean/peak values while the GPU was busy.
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BUSY_THRESHOLD = 10  # % utilization above which the GPU counts as "training"


def load_log(path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    # Header looks like "utilization.gpu [%]"; strip the units from names and values.
    df.columns = [re.sub(r"\s*\[.*\]", "", c).strip() for c in df.columns]
    df = df[df["timestamp"] != "timestamp"]  # repeated headers if the log was restarted
    for c in df.columns:
        if c != "timestamp":
            df[c] = pd.to_numeric(df[c].astype(str).str.extract(r"([-\d.]+)")[0], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    df["minutes"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds() / 60
    return df


def summarize(df: pd.DataFrame) -> dict:
    busy = df[df["utilization.gpu"] > BUSY_THRESHOLD]
    s = {
        "log_duration_min": round(float(df["minutes"].iloc[-1]), 1),
        "busy_duration_min": round(len(busy) * df["minutes"].diff().median(), 1) if len(busy) else 0.0,
        "gpu_util_mean_busy_pct": round(float(busy["utilization.gpu"].mean()), 1) if len(busy) else 0.0,
        "gpu_util_p95_pct": round(float(df["utilization.gpu"].quantile(0.95)), 1),
        "memory_peak_mib": int(df["memory.used"].max()),
        "memory_total_mib": int(df["memory.total"].max()) if "memory.total" in df else None,
    }
    if "power.draw" in df:
        s["power_mean_busy_w"] = round(float(busy["power.draw"].mean()), 1) if len(busy) else 0.0
        s["power_peak_w"] = round(float(df["power.draw"].max()), 1)
        s["energy_wh"] = round(float((df["power.draw"] * df["minutes"].diff().fillna(0) / 60).sum()), 1)
    if "temperature.gpu" in df:
        s["temperature_peak_c"] = int(df["temperature.gpu"].max())
    return s


def plot(df: pd.DataFrame, title: str, path: Path) -> None:
    panels = [("utilization.gpu", "GPU utilization (%)", (0, 105))]
    panels.append(("memory.used", "Memory used (GiB)", None))
    if "power.draw" in df:
        panels.append(("power.draw", "Power (W)", None))
    if "temperature.gpu" in df:
        panels.append(("temperature.gpu", "Temperature (°C)", None))

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.1 * len(panels)), sharex=True)
    smooth = max(1, len(df) // 300)  # light rolling mean so long logs stay readable
    for ax, (col, label, ylim) in zip(axes, panels):
        y = df[col] / 1024 if col == "memory.used" else df[col]
        ax.plot(df["minutes"], y, lw=0.6, alpha=0.35, color="tab:blue")
        ax.plot(df["minutes"], y.rolling(smooth, min_periods=1).mean(), lw=1.4, color="tab:blue")
        if col == "memory.used" and "memory.total" in df:
            ax.axhline(df["memory.total"].max() / 1024, ls="--", lw=0.8, color="gray", label="total")
            ax.legend(loc="lower right", fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Time since start of log (minutes)")
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="CSV written by nvidia-smi --format=csv -l 1")
    ap.add_argument("--out", help="output PNG (default: next to the log)")
    ap.add_argument("--title", default="GPU usage during training (NVIDIA L4)")
    args = ap.parse_args()

    df = load_log(args.log)
    if df.empty:
        raise SystemExit("No rows could be parsed from the log")
    out = Path(args.out) if args.out else Path(args.log).with_suffix(".png")
    plot(df, args.title, out)
    s = summarize(df)
    out.with_name(out.stem + "_summary.json").write_text(json.dumps(s, indent=2))
    print(f"Chart: {out}")
    for k, v in s.items():
        print(f"  {k:<24} {v}")


if __name__ == "__main__":
    main()
