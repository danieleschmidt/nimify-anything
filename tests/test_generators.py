"""Tests for NIMServiceGenerator, OpenAPIGenerator, DockerfileGenerator."""

import json
import tempfile
from pathlib import Path

import pytest

from src.nimify.inspector import ModelInspector
from src.nimify.generator import (
    NIMServiceGenerator,
    OpenAPIGenerator,
    DockerfileGenerator,
)


@pytest.fixture
def meta():
    return ModelInspector("resnet50.onnx", mock=True).inspect()


@pytest.fixture
def tmpdir_path():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# NIMServiceGenerator
# ---------------------------------------------------------------------------

class TestNIMServiceGenerator:
    def test_creates_app_py(self, meta, tmpdir_path):
        path = NIMServiceGenerator(meta).generate(tmpdir_path)
        assert path.exists()
        assert path.name == "app.py"

    def test_app_contains_fastapi(self, meta, tmpdir_path):
        path = NIMServiceGenerator(meta).generate(tmpdir_path)
        content = path.read_text()
        assert "FastAPI" in content

    def test_app_contains_predict_endpoint(self, meta, tmpdir_path):
        path = NIMServiceGenerator(meta).generate(tmpdir_path)
        content = path.read_text()
        assert "/predict" in content

    def test_app_contains_health_endpoint(self, meta, tmpdir_path):
        path = NIMServiceGenerator(meta).generate(tmpdir_path)
        content = path.read_text()
        assert "/health" in content

    def test_app_contains_metrics_endpoint(self, meta, tmpdir_path):
        path = NIMServiceGenerator(meta).generate(tmpdir_path)
        content = path.read_text()
        assert "/metrics" in content

    def test_app_contains_input_tensor_name(self, meta, tmpdir_path):
        path = NIMServiceGenerator(meta).generate(tmpdir_path)
        content = path.read_text()
        assert meta.inputs[0].name in content

    def test_app_contains_output_tensor_name(self, meta, tmpdir_path):
        path = NIMServiceGenerator(meta).generate(tmpdir_path)
        content = path.read_text()
        assert meta.outputs[0].name in content

    def test_app_contains_prometheus_counter(self, meta, tmpdir_path):
        path = NIMServiceGenerator(meta).generate(tmpdir_path)
        content = path.read_text()
        assert "Counter" in content

    def test_app_is_valid_python_syntax(self, meta, tmpdir_path):
        path = NIMServiceGenerator(meta).generate(tmpdir_path)
        import ast
        ast.parse(path.read_text())  # raises SyntaxError if invalid


# ---------------------------------------------------------------------------
# OpenAPIGenerator
# ---------------------------------------------------------------------------

class TestOpenAPIGenerator:
    def test_creates_openapi_json(self, meta, tmpdir_path):
        path = OpenAPIGenerator(meta).generate(tmpdir_path)
        assert path.exists()
        assert path.name == "openapi.json"

    def test_valid_json(self, meta, tmpdir_path):
        path = OpenAPIGenerator(meta).generate(tmpdir_path)
        spec = json.loads(path.read_text())
        assert isinstance(spec, dict)

    def test_openapi_version(self, meta, tmpdir_path):
        path = OpenAPIGenerator(meta).generate(tmpdir_path)
        spec = json.loads(path.read_text())
        assert spec["openapi"].startswith("3.0")

    def test_has_predict_path(self, meta, tmpdir_path):
        path = OpenAPIGenerator(meta).generate(tmpdir_path)
        spec = json.loads(path.read_text())
        assert "/predict" in spec["paths"]

    def test_has_health_path(self, meta, tmpdir_path):
        path = OpenAPIGenerator(meta).generate(tmpdir_path)
        spec = json.loads(path.read_text())
        assert "/health" in spec["paths"]

    def test_has_schemas(self, meta, tmpdir_path):
        path = OpenAPIGenerator(meta).generate(tmpdir_path)
        spec = json.loads(path.read_text())
        assert "PredictRequest" in spec["components"]["schemas"]
        assert "PredictResponse" in spec["components"]["schemas"]

    def test_request_schema_has_input_field(self, meta, tmpdir_path):
        path = OpenAPIGenerator(meta).generate(tmpdir_path)
        spec = json.loads(path.read_text())
        req_schema = spec["components"]["schemas"]["PredictRequest"]
        assert meta.inputs[0].name in req_schema["properties"]

    def test_response_schema_has_output_field(self, meta, tmpdir_path):
        path = OpenAPIGenerator(meta).generate(tmpdir_path)
        spec = json.loads(path.read_text())
        resp_schema = spec["components"]["schemas"]["PredictResponse"]
        assert meta.outputs[0].name in resp_schema["properties"]


# ---------------------------------------------------------------------------
# DockerfileGenerator
# ---------------------------------------------------------------------------

class TestDockerfileGenerator:
    def test_creates_dockerfile(self, meta, tmpdir_path):
        path = DockerfileGenerator(meta).generate(tmpdir_path)
        assert path.exists()
        assert path.name == "Dockerfile"

    def test_has_from(self, meta, tmpdir_path):
        path = DockerfileGenerator(meta).generate(tmpdir_path)
        content = path.read_text()
        assert "FROM" in content

    def test_multi_stage(self, meta, tmpdir_path):
        path = DockerfileGenerator(meta).generate(tmpdir_path)
        content = path.read_text()
        assert content.count("FROM") >= 2  # multi-stage

    def test_has_expose(self, meta, tmpdir_path):
        path = DockerfileGenerator(meta).generate(tmpdir_path)
        content = path.read_text()
        assert "EXPOSE" in content

    def test_has_healthcheck(self, meta, tmpdir_path):
        path = DockerfileGenerator(meta).generate(tmpdir_path)
        content = path.read_text()
        assert "HEALTHCHECK" in content

    def test_non_root_user(self, meta, tmpdir_path):
        path = DockerfileGenerator(meta).generate(tmpdir_path)
        content = path.read_text()
        assert "USER nimuser" in content
