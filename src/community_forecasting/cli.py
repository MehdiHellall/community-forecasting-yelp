"""Command-line interface for portfolio-grade project checks."""

from __future__ import annotations

import argparse
import sys

import nbformat

from community_forecasting.leakage import scan_notebooks_for_static_leakage
from community_forecasting.paths import NOTEBOOKS_DIR, OUTPUTS_DIR, resolve_repo_path
from community_forecasting.validation import load_result_summary, validate_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cf-yelp")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate-outputs")
    validate_parser.add_argument("--outputs-dir", default=None)
    validate_parser.set_defaults(func=validate_outputs_command)

    summary_parser = subparsers.add_parser("summarize-results")
    summary_parser.add_argument("--outputs-dir", default=None)
    summary_parser.set_defaults(func=summarize_results_command)

    leakage_parser = subparsers.add_parser("leakage-check")
    leakage_parser.add_argument("--notebooks-dir", default=None)
    leakage_parser.set_defaults(func=leakage_check_command)

    notebooks_parser = subparsers.add_parser("execute-notebooks")
    notebooks_parser.add_argument("--notebooks-dir", default=None)
    notebooks_parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Parse notebooks and fail on saved error outputs without running the full "
            "Yelp pipeline."
        ),
    )
    notebooks_parser.set_defaults(func=execute_notebooks_command)
    return parser


def validate_outputs_command(args: argparse.Namespace) -> int:
    outputs_dir = resolve_repo_path(args.outputs_dir, OUTPUTS_DIR)
    results = validate_outputs(outputs_dir)
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"{status} {result.name}: {result.message}")
    return 0 if all(result.ok for result in results) else 1


def summarize_results_command(args: argparse.Namespace) -> int:
    outputs_dir = resolve_repo_path(args.outputs_dir, OUTPUTS_DIR)
    summary = load_result_summary(outputs_dir)
    print("Community Forecasting with Yelp Data")
    print(f"Split: {summary['split']}")
    print(
        "Best review-count model: "
        f"{summary['best_review_count_model']} "
        f"(WAPE {float(summary['best_review_count_WAPE']):.3f})"
    )
    print(
        "Best pulse model: "
        f"{summary['best_pulse_model']} "
        f"(F1 {float(summary['best_pulse_F1']):.3f}, "
        f"PR-AUC {float(summary['best_pulse_PR_AUC']):.3f})"
    )
    print(
        "Best top-10% pulse precision: "
        f"{float(summary['best_top10_precision']):.3f} "
        f"with {summary['best_top10_precision_model']}"
    )
    print(f"Takeaway: {summary['modality_takeaway']}")
    return 0


def leakage_check_command(args: argparse.Namespace) -> int:
    notebooks_dir = resolve_repo_path(args.notebooks_dir, NOTEBOOKS_DIR)
    findings = scan_notebooks_for_static_leakage(notebooks_dir)
    if not findings:
        print("PASS leakage-check: no forbidden snapshot feature sets found")
        return 0
    for finding in findings:
        print(f"{finding.severity.upper()} leakage-check: {finding.message}")
    return 1


def execute_notebooks_command(args: argparse.Namespace) -> int:
    if not args.smoke:
        print(
            "Use --smoke for CI-safe notebook validation. Full execution requires local Yelp data."
        )
        return 2
    notebooks_dir = resolve_repo_path(args.notebooks_dir, NOTEBOOKS_DIR)
    notebooks = sorted(notebooks_dir.glob("*.ipynb"))
    if not notebooks:
        print(f"FAIL execute-notebooks --smoke: no notebooks found in {notebooks_dir}")
        return 1

    failed = False
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4)
        error_outputs = [
            output
            for cell in notebook.cells
            if cell.get("cell_type") == "code"
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        if error_outputs:
            failed = True
            print(f"FAIL {path.name}: {len(error_outputs)} saved error outputs")
        else:
            print(f"PASS {path.name}: parsed {len(notebook.cells)} cells")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
