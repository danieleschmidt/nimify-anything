"""Tests for the nimify CLI (end-to-end with click test runner)."""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from src.nimify.cli import main


@pytest.fixture
def runner():
    return CliRunner()


class TestCLI:
    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "ONNX" in result.output or "model" in result.output.lower()

    def test_mock_run(self, runner):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(main, ["model.onnx", "--output", tmpdir, "--mock"])
            assert result.exit_code == 0, result.output
            assert "Done" in result.output or "✔" in result.output

    def test_generates_all_artifacts(self, runner):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(main, ["model.onnx", "--output", tmpdir, "--mock"])
            assert result.exit_code == 0, result.output
            files = {p.name for p in Path(tmpdir).iterdir()}
            assert "app.py" in files
            assert "openapi.json" in files
            assert "Dockerfile" in files

    def test_openapi_json_valid(self, runner):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner.invoke(main, ["model.onnx", "--output", tmpdir, "--mock"])
            spec = json.loads((Path(tmpdir) / "openapi.json").read_text())
            assert spec["openapi"].startswith("3.0")

    def test_nonexistent_model_without_mock(self, runner):
        """Without --mock, a missing file still returns mock metadata (fallback)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(main, ["/tmp/ghost_model_xyz.onnx", "--output", tmpdir])
            assert result.exit_code == 0
