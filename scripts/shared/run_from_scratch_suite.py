from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_FAMILIES = ["oxide", "nitride"]
DEFAULT_NS = [50, 500]
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
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def update_batch_size(config_path: Path, batch_size: int) -> None:
    payload = load_json(config_path)
    payload["batch_size"] = batch_size
    write_json(config_path, payload)


def log_has_cuda_oom(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    tail = log_path.read_text(encoding="utf-8", errors="ignore")[-20000:].lower()
    return any(pattern in tail for pattern in OOM_STRINGS)


def run_train_with_oom_retry(
    python: str,
    repo: Path,
    config_path: Path,
    output_dir: Path,
    dataset_root: Path,
    device: str,
    log_path: Path,
    run_name: str,
    min_batch_size: int,
) -> int:
    batch_size = int(load_json(config_path)["batch_size"])
    while True:
        train_cmd = [
            python,
            "scripts/shared/train_alignn_from_scratch.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--dataset-root",
            str(dataset_root),
            "--device",
            device,
        ]
        try:
            run_command(train_cmd, repo, log_path=log_path)
            return batch_size
        except subprocess.CalledProcessError:
            if not log_has_cuda_oom(log_path):
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
    remote_url = subprocess.run(
        ["git", "remote", "get-url", remote],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if "$GITHUB_TOKEN" in remote_url or "%24GITHUB_TOKEN" in remote_url:
        raise RuntimeError(
            "Git remote URL still contains a literal $GITHUB_TOKEN placeholder. "
            "Re-set GITHUB_TOKEN in Colab and run "
            "'git remote set-url origin \"https://${GITHUB_TOKEN}@github.com/TheArchitect999/domain_shift-alignn-domain-shift.git\"' "
            "before retrying."
        )
    subprocess.run(["git", "commit", "-m", message], cwd=repo, check=True)
    subprocess.run(["git", "push", remote, f"HEAD:{branch}"], cwd=repo, check=True)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    parser.add_argument("--Ns", nargs="+", type=int, default=DEFAULT_NS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--link-mode", choices=["copy", "hardlink", "symlink"], default="hardlink")
    parser.add_argument("--run-subdir", default="train_alignn_from_scratch")
    parser.add_argument("--config-dir", default="configs/protocol_2/from_scratch")
    parser.add_argument("--report-dir", default="results/summaries/protocol_2/From Scratch")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--git-push-after-run", action="store_true")
    parser.add_argument("--git-remote", default="origin")
    parser.add_argument("--git-branch", default=DEFAULT_BRANCH)
    parser.add_argument(
        "--pretrained-config",
        default="configs/pretrained/config.json",
    )
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    python = sys.executable
    config_dir = repo / args.config_dir
    report_dir = repo / args.report_dir

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
                output_dir = run_root / args.run_subdir
                summary_path = output_dir / "summary.json"
                config_path = config_dir / f"{family}_from_scratch_N{n_value}_seed{seed}.json"
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
                        print(json.dumps({"event": "skip_completed", "run": run_name, "summary": str(summary_path)}))
                        continue

                    final_batch_size = run_train_with_oom_retry(
                        python=python,
                        repo=repo,
                        config_path=config_path,
                        output_dir=output_dir,
                        dataset_root=dataset_root,
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
                    if args.git_push_after_run:
                        commit_message = (
                            f"colab: complete from_scratch from_scratch "
                            f"{family} N{n_value} seed{seed}"
                        )
                        git_commit_and_push(
                            repo,
                            remote=args.git_remote,
                            branch=args.git_branch,
                            message=commit_message,
                            paths=[dataset_root, output_dir, config_path],
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
        "requested_runs": total,
        "completed_runs": completed,
        "skipped_existing_runs": skipped,
        "failed_runs": failures,
        "run_subdir": args.run_subdir,
        "config_dir": str(config_dir),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_out = report_dir / "run_suite_summary.json"
    summary_out.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({"event": "suite_finished", **aggregate}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
