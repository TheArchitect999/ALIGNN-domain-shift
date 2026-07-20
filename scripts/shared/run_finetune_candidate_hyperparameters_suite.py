from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_NS = [10, 50, 100, 200, 500, 1000]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_FAMILIES = ["oxide", "nitride"]
DEFAULT_TAG = "finetune_alignn_hyperparameters_new_script"
DEFAULT_RESULTS_ROOT = "results/protocol_1"


def run_command(cmd: list[str], cwd: Path, log_path: Path | None = None) -> None:
    env = os.environ.copy()
    env.setdefault("DGLBACKEND", "pytorch")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
    env.setdefault("FC_CACHEDIR", "/tmp/fontconfig")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(env["FC_CACHEDIR"]).mkdir(parents=True, exist_ok=True)
    if log_path is None:
        subprocess.run(cmd, cwd=cwd, check=True, env=env)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {' '.join(cmd)}\n")
        handle.flush()
        subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        handle.write("\n")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--experiment-tag", default=DEFAULT_TAG)
    parser.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    parser.add_argument("--Ns", nargs="+", type=int, default=DEFAULT_NS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--link-mode", choices=["copy", "hardlink", "symlink"], default="hardlink")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--pretrained-checkpoint",
        default="models/pretrained/checkpoint_300.pt",
    )
    parser.add_argument(
        "--pretrained-config",
        default="configs/pretrained/config.json",
    )
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    python = sys.executable
    tag = args.experiment_tag
    run_subdir = f"finetune_last2_{tag}"
    config_dir = repo / "configs" / tag
    report_dir = repo / "results" / "derived_evidence" / tag

    failures: list[dict] = []
    completed = 0
    skipped = 0
    total = len(args.families) * len(args.Ns) * len(args.seeds)

    for family in args.families:
        for n_value in args.Ns:
            for seed in args.seeds:
                started_at = time.time()
                run_name = f"{family}:N{n_value}:seed{seed}"
                run_root = repo / args.results_root / family / f"N{n_value}_seed{seed}"
                dataset_root = run_root / "dataset_root"
                split_manifest_path = dataset_root / "split_manifest.json"
                output_dir = run_root / run_subdir
                summary_path = output_dir / "summary.json"
                config_path = config_dir / f"{family}_{tag}_N{n_value}_seed{seed}.finetune_last2.json"
                log_path = output_dir / "run.log"

                print(json.dumps({"event": "start", "run": run_name, "output_dir": str(output_dir)}))

                try:
                    if args.force or not split_manifest_path.exists():
                        prep_cmd = [
                            python,
                            "scripts/shared/prepare_baseline_finetune_dataset.py",
                            "--family",
                            family,
                            "--N",
                            str(n_value),
                            "--seed",
                            str(seed),
                            "--repo-root",
                            ".",
                            "--results-root",
                            args.results_root,
                            "--link-mode",
                            args.link_mode,
                        ]
                        run_command(prep_cmd, repo)

                    split_manifest = load_json(split_manifest_path)
                    config_cmd = [
                        python,
                        "scripts/shared/write_baseline_alignn_config.py",
                        "--dataset-root",
                        str(dataset_root),
                        "--out",
                        str(config_path),
                        "--n-train",
                        str(split_manifest["n_train"]),
                        "--n-val",
                        str(split_manifest["n_val"]),
                        "--n-test",
                        str(split_manifest["n_test"]),
                        "--epochs",
                        str(args.epochs),
                        "--batch-size",
                        str(args.batch_size),
                        "--lr",
                        str(args.lr),
                        "--seed",
                        str(seed),
                        "--pretrained-config",
                        str((repo / args.pretrained_config).resolve()),
                    ]
                    run_command(config_cmd, repo)

                    if summary_path.exists() and not args.force:
                        skipped += 1
                        print(
                            json.dumps(
                                {
                                    "event": "skip_completed",
                                    "run": run_name,
                                    "summary_path": str(summary_path),
                                    "elapsed_seconds": time.time() - started_at,
                                }
                            )
                        )
                        continue

                    finetune_cmd = [
                        python,
                        "scripts/shared/fine_tune_last_two_alignn_layers_candidate_hyperparameters.py",
                        "--config",
                        str(config_path),
                        "--output-dir",
                        str(output_dir),
                        "--dataset-root",
                        str(dataset_root),
                        "--pretrained-checkpoint",
                        str((repo / args.pretrained_checkpoint).resolve()),
                        "--pretrained-config",
                        str((repo / args.pretrained_config).resolve()),
                        "--device",
                        args.device,
                    ]
                    run_command(finetune_cmd, repo, log_path=log_path)
                    summary = load_json(summary_path)
                    completed += 1
                    print(
                        json.dumps(
                            {
                                "event": "completed",
                                "run": run_name,
                                "test_mae_eV_per_atom": summary["test_mae_eV_per_atom"],
                                "elapsed_seconds": time.time() - started_at,
                            }
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    failure = {
                        "run": run_name,
                        "error": repr(exc),
                        "elapsed_seconds": time.time() - started_at,
                    }
                    failures.append(failure)
                    print(json.dumps({"event": "failed", **failure}))
                    if not args.continue_on_error:
                        raise

    aggregate = {
        "experiment_tag": tag,
        "run_subdir": run_subdir,
        "requested_runs": total,
        "completed_runs": completed,
        "skipped_existing_runs": skipped,
        "failed_runs": failures,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_out = report_dir / "run_suite_summary.json"
    summary_out.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({"event": "suite_finished", **aggregate}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
