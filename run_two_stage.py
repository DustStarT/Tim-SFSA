#!/usr/bin/env python3
"""Convenience launcher for classification-only and two-stage training."""

import argparse
import sys

from configs.default_config import get_config
from main import run_classification_experiment, run_two_stage_experiment


def build_config(args):
    cfg = get_config()

    if args.mode == "classification":
        cfg.run_mode = "classification_only"
        cfg.model.two_stage.enabled = True
        cfg.model.two_stage.classification_stage.enabled = True
        cfg.model.two_stage.survival_stage.enabled = False
    else:
        cfg.run_mode = "two_stage"
        cfg.model.two_stage.enabled = True
        cfg.model.two_stage.classification_stage.enabled = True
        cfg.model.two_stage.survival_stage.enabled = True

    if args.device == "auto":
        import torch

        cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        cfg.training.device = args.device

    cfg.model.two_stage.classification_stage.num_epochs = args.epochs
    cfg.model.two_stage.classification_stage.batch_size = args.batch_size
    cfg.model.two_stage.classification_stage.learning_rate = args.lr

    if args.mode == "two_stage":
        cfg.model.two_stage.survival_stage.freeze_lstm = args.freeze_lstm
        cfg.model.two_stage.survival_stage.fine_tune_epochs = args.fine_tune_epochs
        cfg.model.two_stage.survival_stage.fine_tune_lr = args.lr * 0.1

    if args.output_dir:
        cfg.results_dir = args.output_dir

    return cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run classification-only or two-stage training for FlameShadowModel."
    )
    parser.add_argument(
        "--mode",
        choices=["classification", "two_stage"],
        default="two_stage",
        help="Execution mode.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Classification epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate.")
    parser.add_argument(
        "--freeze_lstm",
        action="store_true",
        help="Freeze the sequence encoder during the survival fine-tuning stage.",
    )
    parser.add_argument(
        "--fine_tune_epochs",
        type=int,
        default=30,
        help="Fine-tuning epochs for the survival stage.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional override for config.results_dir.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: cuda, cpu, or auto.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_config(args)

    print(f"Mode: {args.mode}")
    print(f"Device: {cfg.training.device}")
    print(f"Classification epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    if args.mode == "two_stage":
        print(f"Freeze encoder: {args.freeze_lstm}")
        print(f"Fine-tune epochs: {args.fine_tune_epochs}")
    print(f"Output dir: {cfg.results_dir}")
    print("-" * 50)

    try:
        if args.mode == "classification":
            results = run_classification_experiment(cfg)
            print(f"Classification finished. Best val AUC: {results.get('best_val_auc', 'N/A')}")
        else:
            results = run_two_stage_experiment(cfg)
            print("Two-stage training finished.")
            print(
                "Classification AUC: "
                f"{results.get('classification_results', {}).get('best_val_auc', 'N/A')}"
            )
            print(
                "Survival C-index: "
                f"{results.get('survival_metrics', {}).get('c_index', 'N/A')}"
            )
        return True
    except Exception as exc:
        print(f"Run failed: {exc}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
