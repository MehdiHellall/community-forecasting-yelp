from __future__ import annotations

import nbformat
import pandas as pd

from community_forecasting.cli import main
from community_forecasting.validation import load_result_summary, validate_outputs


def test_validate_outputs_passes_committed_outputs():
    results = validate_outputs("outputs")

    assert results
    assert all(result.ok for result in results)


def test_load_result_summary_reads_single_row():
    summary = load_result_summary("outputs")

    assert summary["split"] == "normal_pre_covid_test"
    assert "WAPE" not in summary["modality_takeaway"]


def test_summarize_results_cli_prints_key_metrics(capsys):
    exit_code = main(["summarize-results", "--outputs-dir", "outputs"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Best review-count model" in captured.out
    assert "Best pulse model" in captured.out


def test_validate_outputs_detects_missing_columns(tmp_path):
    for filename in [
        "model_comparison_summary.csv",
        "forecasting_metrics.csv",
        "attention_pulse_metrics.csv",
        "attention_pulse_topk_metrics.csv",
        "attention_pulse_calibration.csv",
        "attention_pulse_case_studies.csv",
        "nlp_tfidf_terms.csv",
        "pulse_predecessor_analysis.csv",
    ]:
        pd.DataFrame({"wrong": [1]}).to_csv(tmp_path / filename, index=False)
    for directory in ["eda", "feature_engineering", "models", "sna"]:
        (tmp_path / "figures" / directory).mkdir(parents=True)

    results = validate_outputs(tmp_path)

    assert any(not result.ok and "missing columns" in result.message for result in results)


def test_validate_outputs_cli_returns_nonzero_for_missing_files(tmp_path, capsys):
    exit_code = main(["validate-outputs", "--outputs-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "model_comparison_summary.csv" in captured.out


def test_leakage_check_cli_reports_forbidden_notebook_feature_set(tmp_path, capsys):
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell('business_features = [\n    "business_review_count",\n]\n')
        ]
    )
    nbformat.write(notebook, tmp_path / "leaky.ipynb")

    exit_code = main(["leakage-check", "--notebooks-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "business_review_count" in captured.out


def test_leakage_check_cli_reports_stale_notebook_outputs(tmp_path, capsys):
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_markdown_cell("clean source")],
        metadata={},
    )
    nbformat.write(notebook, tmp_path / "stale.ipynb")
    notebook_path = tmp_path / "stale.ipynb"
    text = notebook_path.read_text(encoding="utf-8")
    notebook_path.write_text(text.replace("clean source", "<td>business_review_count</td>"))

    exit_code = main(["leakage-check", "--notebooks-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "stale notebook output" in captured.out


def test_execute_notebooks_requires_smoke_flag(capsys):
    exit_code = main(["execute-notebooks"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Use --smoke" in captured.out


def test_execute_notebooks_smoke_parses_clean_notebook(tmp_path, capsys):
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell("hello")])
    nbformat.write(notebook, tmp_path / "clean.ipynb")

    exit_code = main(["execute-notebooks", "--smoke", "--notebooks-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PASS clean.ipynb" in captured.out


def test_execute_notebooks_smoke_fails_on_saved_error_output(tmp_path, capsys):
    cell = nbformat.v4.new_code_cell("raise ValueError('boom')")
    cell.outputs = [
        nbformat.v4.new_output(
            "error",
            ename="ValueError",
            evalue="boom",
            traceback=["ValueError: boom"],
        )
    ]
    notebook = nbformat.v4.new_notebook(cells=[cell])
    nbformat.write(notebook, tmp_path / "error.ipynb")

    exit_code = main(["execute-notebooks", "--smoke", "--notebooks-dir", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL error.ipynb" in captured.out
