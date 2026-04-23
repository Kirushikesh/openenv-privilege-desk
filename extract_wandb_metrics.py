#!/usr/bin/env python3
"""
Extract metrics from W&B training runs and generate graphs.
Usage:
  python extract_wandb_metrics.py --entity YOUR_ENTITY --project YOUR_PROJECT --output ./metrics_report
"""

import json
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import wandb
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install wandb pandas matplotlib python-dotenv")
    exit(1)


def fetch_run_metrics(run_id: str, entity: str, project: str) -> Dict:
    """Fetch all metrics from a single W&B run."""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # Extract history (timeseries metrics)
    history = run.history(samples=10000)  # Large sample size to get all steps

    return {
        "run_id": run.id,
        "run_name": run.name,
        "state": run.state,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
        "summary": run.summary,
        "history": history,
        "config": dict(run.config),
    }


def extract_metric_columns(history_df: pd.DataFrame, debug: bool = False) -> Dict[str, List[float]]:
    """Extract key training metrics from history."""
    metrics = {}

    if debug:
        print(f"\n📋 Available columns in history: {list(history_df.columns)}")

    # Primary reward signals (W&B naming convention)
    reward_cols = {
        "train/rewards/reward_episode_score/mean": "Episode Score",
        "train/rewards/reward_step_efficiency/mean": "Step Efficiency",
        "train/rewards/reward_format/mean": "Format Reward",
        "train/rewards/reward_efficiency/mean": "Efficiency Reward",
        "train/reward": "Total Reward",
    }

    # Training health
    loss_cols = {
        "train/loss": "Loss",
        "train/kl": "KL Divergence",
        "train/grad_norm": "Grad Norm",
        "train/learning_rate": "Learning Rate",
    }

    # Extract reward signals
    for wb_col, display_name in reward_cols.items():
        if wb_col in history_df.columns:
            values = history_df[wb_col].dropna().tolist()
            if values:
                metrics[display_name] = values

    # Extract training metrics
    for wb_col, display_name in loss_cols.items():
        if wb_col in history_df.columns:
            values = history_df[wb_col].dropna().tolist()
            if values:
                metrics[display_name] = values

    return metrics


def plot_rewards(metrics: Dict[str, List[float]], output_dir: Path):
    """Plot primary reward signals."""
    reward_keys = ["Episode Score", "Step Efficiency", "Format Reward", "Efficiency Reward", "Total Reward"]
    available_rewards = [k for k in reward_keys if k in metrics]

    if not available_rewards:
        print("⚠️  No reward metrics found")
        return

    plt.figure(figsize=(12, 6))
    for key in available_rewards:
        values = metrics[key]
        plt.plot(range(len(values)), values, label=key, marker="o", markersize=3)

    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Training Rewards Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = output_dir / "rewards.png"
    plt.savefig(output_file, dpi=150)
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_training_health(metrics: Dict[str, List[float]], output_dir: Path):
    """Plot loss, KL divergence, gradient norm."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    if "Loss" in metrics:
        axes[0, 0].plot(metrics["Loss"], label="Loss", color="tab:red")
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

    # KL divergence
    if "KL Divergence" in metrics:
        axes[0, 1].plot(metrics["KL Divergence"], label="KL Divergence", color="tab:orange")
        axes[0, 1].set_title("KL Divergence (should stay < 0.3)")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].axhline(y=0.3, color="r", linestyle="--", alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()

    # Gradient norm
    if "Grad Norm" in metrics:
        axes[1, 0].plot(metrics["Grad Norm"], label="Grad Norm", color="tab:green")
        axes[1, 0].set_title("Gradient Norm (should be < 5)")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].axhline(y=5, color="r", linestyle="--", alpha=0.5)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

    # Learning rate
    if "Learning Rate" in metrics:
        axes[1, 1].plot(metrics["Learning Rate"], label="LR", color="tab:blue")
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

    plt.tight_layout()
    output_file = output_dir / "training_health.png"
    plt.savefig(output_file, dpi=150)
    print(f"✓ Saved: {output_file}")
    plt.close()


def save_metrics_csv(metrics: Dict[str, List[float]], output_dir: Path):
    """Save metrics to CSV for spreadsheet analysis."""
    # Find the maximum length
    max_len = max(len(v) for v in metrics.values()) if metrics else 0

    # Pad all to same length
    padded = {}
    for key, values in metrics.items():
        padded[key] = values + [None] * (max_len - len(values))

    df = pd.DataFrame(padded)
    output_file = output_dir / "metrics.csv"
    df.to_csv(output_file, index=False)
    print(f"✓ Saved: {output_file}")

    # Also print summary stats
    print("\n📊 Summary Statistics:")
    print(df.describe().to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Extract W&B training metrics and generate graphs"
    )
    parser.add_argument("--entity", required=True, help="W&B entity (username or team)")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--run-id", help="Specific run ID (if not provided, uses latest)")
    parser.add_argument("--output", default="./metrics_report", help="Output directory")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🔗 Connecting to W&B: {args.entity}/{args.project}")

    api = wandb.Api()

    # Get runs
    if args.run_id:
        runs = [api.run(f"{args.entity}/{args.project}/{args.run_id}")]
        print(f"📌 Fetching specific run: {args.run_id}")
    else:
        # Get all completed runs, sorted by creation time (newest first)
        runs = api.runs(f"{args.entity}/{args.project}", {"state": "finished"})
        runs = sorted(runs, key=lambda r: r.created_at, reverse=True)
        print(f"📌 Found {len(runs)} completed runs. Using latest...")
        if runs:
            runs = [runs[0]]
        else:
            print("❌ No completed runs found!")
            return

    for run in runs:
        print(f"\n🏃 Run: {run.name} (ID: {run.id})")
        print(f"   State: {run.state}")
        print(f"   Steps: {len(run.history())}")

        # Fetch metrics
        history = run.history(samples=10000)
        metrics = extract_metric_columns(history, debug=False)

        if not metrics:
            print("   ⚠️  No metrics found in this run")
            continue

        print(f"   ✓ Extracted {len(metrics)} metric types")

        # Generate outputs
        print("\n   Generating visualizations...")
        plot_rewards(metrics, output_dir)
        plot_training_health(metrics, output_dir)
        save_metrics_csv(metrics, output_dir)

        # Save metadata
        metadata = {
            "run_id": run.id,
            "run_name": run.name,
            "created_at": str(run.created_at),
            "state": run.state,
            "config": dict(run.config),
            "summary": {k: float(v) if isinstance(v, (int, float)) else str(v)
                       for k, v in run.summary.items()},
        }

        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✓ Saved metadata: {metadata_file}")

    print(f"\n✅ All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
