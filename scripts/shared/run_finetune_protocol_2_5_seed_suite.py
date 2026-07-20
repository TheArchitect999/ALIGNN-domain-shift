from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_TAG = "finetune"
DEFAULT_FAMILIES = ["oxide", "nitride"]
DEFAULT_NS = [10, 50, 100, 200, 500, 1000]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_BRANCH = "main"
DEFAULT_RESULTS_ROOT = "results/protocol_2"
OOM_STRINGS = (
    "cuda error: out of memory",
    "cuda out of memory",
    "cublas_status_alloc_failed",
)


def run_command(cmd: list[str], cwd: Path, log_path: Path | None = None) -> None:
    env = os.environ.copy()
    env.setdefault("DGLBACKEND", "pytorch")
    temp_root = Path(tempfile.gettempdir())
    env.setdefault("MPLCONFIGDIR", str(temp_root / "matplotlib"))
    env.setdefault("XDG_CACHE_HOME", str(temp_root / "xdg-cache"))
    env.setdefault("FC_CACHEDIR", str(temp_root / "fontconfig"))
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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def update_batch_size(config_path: Path, batch_size: int) -> None:
    payload = load_json(config_path)
    payload["batch_size"] = batch_size
    write_json(config_path, payload)


def log_has_cuda_oom(log_path: Path, start_offset: int = 0) -> bool:
    if not log_path.exists():
        return False
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        handle.seek(start_offset)
        tail = handle.read()[-20000:].lower()
    return any(pattern in tail for pattern in OOM_STRINGS)


def run_train_with_oom_retry(
    python: str,
    repo: Path,
    config_path: Path,
    output_dir: Path,
    dataset_root: Path,
    checkpoint_path: Path,
    pretrained_config_path: Path,
    device: str,
    log_path: Path,
    run_name: str,
    min_batch_size: int,
) -> int:
    batch_size = int(load_json(config_path)["batch_size"])
    while True:
        attempt_start = log_path.stat().st_size if log_path.exists() else 0
        finetune_cmd = [
            python,
            "scripts/shared/fine_tune_last_two_alignn_layers.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--dataset-root",
            str(dataset_root),
            "--pretrained-checkpoint",
            str(checkpoint_path),
            "--pretrained-config",
            str(pretrained_config_path),
            "--device",
            device,
        ]
        try:
            run_command(finetune_cmd, repo, log_path=log_path)
            return batch_size
        except subprocess.CalledProcessError:
            if not log_has_cuda_oom(log_path, start_offset=attempt_start):
                raise
            if batch_size <= min_batch_size:
                raise
            next_batch_size = max(min_batch_size, batch_size // 2)
            if next_batch_size == batch_size:
                raise
            print(
                json.dumps(
                    {
                        "event": "retry_after_cuda_oom",
                        "run": run_name,
                        "old_batch_size": batch_size,
                        "new_batch_size": next_batch_size,
                    }
                )
            )
            update_batch_size(config_path, next_batch_size)
            batch_size = next_batch_size


def update_progress_manifest(
    progress_path: Path,
    *,
    experiment_tag: str,
    run_subdir: str,
    config_dir: Path,
    family: str,
    n_value: int,
    seed: int,
    batch_size: int,
    summary_path: Path,
    test_mae: float,
) -> dict:
    payload = (
        load_json(progress_path)
        if progress_path.exists()
        else {
            "experiment_tag": experiment_tag,
            "run_subdir": run_subdir,
            "config_dir": str(config_dir),
            "completed_runs": [],
            "updated_at_epoch_seconds": None,
        }
    )
    payload["completed_runs"] = [
        row for row in payload.get("completed_runs", [])
        if not (row["family"] == family and row["N"] == n_value and row["seed"] == seed)
    ]
    payload["completed_runs"].append(
        {
            "family": family,
            "N": n_value,
            "seed": seed,
            "final_batch_size": batch_size,
            "test_mae_eV_per_atom": test_mae,
            "summary_path": str(summary_path.resolve()),
        }
    )
    payload["completed_runs"] = sorted(
        payload["completed_runs"],
        key=lambda row: (row["family"], row["N"], row["seed"]),
    )
    payload["updated_at_epoch_seconds"] = time.time()
    write_json(progress_path, payload)
    return payload


def git_commit_and_push(
    repo: Path,
    *,
    remote: str,
    branch: str,
    message: str,
    paths: list[Path],
) -> bool:
    path_args = [str(path) for path in paths]
    subprocess.run(["git", "add", "--", *path_args], cwd=repo, check=True)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *path_args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    if not status.stdout.strip():
        return False
    subprocess.run(["git", "commit", "-m", message], cwd=repo, check=True)
    subprocess.run(["git", "push", remote, f"HEAD:{branch}"], cwd=repo, check=True)
    return True


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
    parser.add_argument("--min-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--link-mode", choices=["copy", "hardlink", "symlink"], default="hardlink"
    )
    parser.add_argument("--run-subdir", default=None)
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--git-push-after-run", action="store_true")
    parser.add_argument("--git-remote", default="origin")
    parser.add_argument("--git-branch", default=DEFAULT_BRANCH)
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
    if tag == "finetune":
        default_run_subdir = "finetune_last2"
        default_config_dir = "configs/protocol_2/finetune"
        default_report_dir = "results/summaries/protocol_2/finetune"
    else:
        default_run_subdir = f"finetune_last2_{tag}"
        default_config_dir = f"configs/{tag}"
        default_report_dir = f"results/derived_evidence/{tag}"
    run_subdir = args.run_subdir or default_run_subdir
    config_dir = repo / (args.config_dir or default_config_dir)
    report_dir = repo / (args.report_dir or default_report_dir)
    progress_path = report_dir / "progress_manifest.json"
    checkpoint_path = (repo / args.pretrained_checkpoint).resolve()
    pretrained_config_path = (repo / args.pretrained_config).resolve()

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
                config_path = (
                    config_dir / f"{family}_{tag}_N{n_value}_seed{seed}.finetune_last2.json"
                )
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
                        str(pretrained_config_path),
                    ]
                    run_command(config_cmd, repo)

                    if summary_path.exists() and not args.force:
                        summary = load_json(summary_path)
                        update_progress_manifest(
                            progress_path,
                            experiment_tag=tag,
                            run_subdir=run_subdir,
                            config_dir=config_dir,
                            family=family,
                            n_value=n_value,
                            seed=seed,
                            batch_size=int(summary.get("final_batch_size", summary["batch_size"])),
                            summary_path=summary_path,
                            test_mae=summary["test_mae_eV_per_atom"],
                        )
                        skipped += 1
                        print(
                            json.dumps(
                                {
                                    "event": "skip_completed",
                                    "run": run_name,
                                    "batch_size": summary.get("final_batch_size", summary["batch_size"]),
                                    "test_mae_eV_per_atom": summary["test_mae_eV_per_atom"],
                                    "summary_path": str(summary_path),
                                    "elapsed_seconds": time.time() - started_at,
                                }
                            )
                        )
                        continue

                    final_batch_size = run_train_with_oom_retry(
                        python=python,
                        repo=repo,
                        config_path=config_path,
                        output_dir=output_dir,
                        dataset_root=dataset_root,
                        checkpoint_path=checkpoint_path,
                        pretrained_config_path=pretrained_config_path,
                        device=args.device,
                        log_path=log_path,
                        run_name=run_name,
                        min_batch_size=args.min_batch_size,
                    )
                    summary = load_json(summary_path)
                    completed += 1
                    print(
                        json.dumps(
                            {
                                "event": "completed",
                                "run": run_name,
                                "batch_size": final_batch_size,
                                "test_mae_eV_per_atom": summary["test_mae_eV_per_atom"],
                                "elapsed_seconds": time.time() - started_at,
                            }
                        )
                    )

                    update_progress_manifest(
                        progress_path,
                        experiment_tag=tag,
                        run_subdir=run_subdir,
                        config_dir=config_dir,
                        family=family,
                        n_value=n_value,
                        seed=seed,
                        batch_size=final_batch_size,
                        summary_path=summary_path,
                        test_mae=summary["test_mae_eV_per_atom"],
                    )
                    if args.git_push_after_run:
                        commit_message = (
                            f"colab: complete {tag} {family} N{n_value} seed{seed}"
                        )
                        git_commit_and_push(
                            repo,
                            remote=args.git_remote,
                            branch=args.git_branch,
                            message=commit_message,
                            paths=[dataset_root, output_dir, config_path, progress_path],
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
