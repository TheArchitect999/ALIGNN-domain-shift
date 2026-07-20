from __future__ import annotations

import argparse
import subprocess
import sys

DEFAULT_FAMILIES = ["oxide", "nitride"]
DEFAULT_NS = [50, 500]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_RUN_SUBDIR = "train_alignn_from_scratch_epochs100_bs32_lr5e5"
DEFAULT_CONFIG_DIR = "configs/protocol_3/from_scratch"
DEFAULT_REPORT_DIR = "results/derived_evidence/from_scratch_epochs100_bs32_lr5e5"
DEFAULT_RESULTS_ROOT = "results/protocol_3"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-subdir", default=DEFAULT_RUN_SUBDIR)
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    parser.add_argument("--Ns", nargs="+", type=int, default=DEFAULT_NS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--link-mode", choices=["copy", "hardlink", "symlink"], default="hardlink"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--git-push-after-run", action="store_true")
    parser.add_argument("--git-remote", default="origin")
    parser.add_argument("--git-branch", default="main")
    parser.add_argument(
        "--pretrained-config",
        default="configs/pretrained/config.json",
    )
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "scripts/shared/run_from_scratch_suite.py",
        "--repo-root",
        args.repo_root,
        "--results-root",
        args.results_root,
        "--run-subdir",
        args.run_subdir,
        "--config-dir",
        args.config_dir,
        "--report-dir",
        args.report_dir,
        "--families",
        *args.families,
        "--Ns",
        *[str(value) for value in args.Ns],
        "--seeds",
        *[str(value) for value in args.seeds],
        "--device",
        args.device,
        "--epochs",
        "100",
        "--batch-size",
        "32",
        "--min-batch-size",
        "32",
        "--lr",
        "0.00005",
        "--link-mode",
        args.link_mode,
        "--pretrained-config",
        args.pretrained_config,
    ]
    if args.force:
        cmd.append("--force")
    if args.continue_on_error:
        cmd.append("--continue-on-error")
    if args.git_push_after_run:
        cmd.extend(
            [
                "--git-push-after-run",
                "--git-remote",
                args.git_remote,
                "--git-branch",
                args.git_branch,
            ]
        )

    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
