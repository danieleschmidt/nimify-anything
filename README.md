# nimify-anything

> Wrap any ONNX model into an NVIDIA NIM-style microservice with a single command.

```
nimify model.onnx --output ./service/
```

Generates a fully-wired FastAPI app, an OpenAPI 3.0 spec, and an optimised multi-stage Dockerfile — all derived directly from the model's input/output signatures.

---

## What it does

`nimify` reads your ONNX model metadata (tensor names, dtypes, shapes) and generates three artefacts:

| File | Description |
|------|-------------|
| `app.py` | FastAPI application with `/predict`, `/health`, `/ready`, `/metrics` endpoints |
| `openapi.json` | OpenAPI 3.0 spec with correct request/response schemas |
| `Dockerfile` | Optimised multi-stage build with non-root user and health-check |

The generated service follows the [NVIDIA NIM](https://docs.nvidia.com/nim/) microservice pattern:

- **POST `/predict`** — inference endpoint with Pydantic-validated input/output matching your model's tensor shapes
- **GET `/health`** — liveness probe
- **GET `/ready`** — readiness probe
- **GET `/metrics`** — Prometheus metrics (request counter + latency histogram)

---

## Installation

```bash
pip install nimify-anything
```

Or from source:

```bash
git clone https://github.com/danieleschmidt/nimify-anything
cd nimify-anything
pip install -e .
```

**Optional — ONNX inspection** (uses `onnx` library for richer metadata):

```bash
pip install "nimify-anything[onnx]"
```

Without the `onnx` extra, nimify falls back to raw protobuf parsing, which works for standard ONNX models.

---

## Usage

```bash
# Real ONNX model
nimify resnet50.onnx --output ./resnet-service/

# Demo / dry-run without a real model
nimify model.onnx --output ./demo/ --mock
```

Then run the service:

```bash
cd ./resnet-service/
pip install fastapi uvicorn prometheus-client pydantic onnxruntime
uvicorn app:app --reload
```

Or with Docker:

```bash
docker build -t resnet-nim .
docker run -p 8000:8000 resnet-nim
```

Test it:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
```

---

## Architecture

```
nimify-anything/
├── src/nimify/
│   ├── inspector.py    # ModelInspector — reads ONNX metadata
│   ├── generator.py    # NIMServiceGenerator, OpenAPIGenerator, DockerfileGenerator
│   └── cli.py          # Click CLI entry point
└── tests/
    ├── test_inspector.py
    ├── test_generators.py
    └── test_cli.py
```

### ModelInspector

Reads ONNX model metadata with a three-tier strategy:

1. **`onnx` library** — full fidelity if installed
2. **Raw protobuf parsing** — stdlib-only fallback, works for standard models
3. **Mock** — deterministic ResNet-50-style metadata for demos/tests

### NIMServiceGenerator

Renders a FastAPI application from model metadata:

- `PredictRequest` / `PredictResponse` Pydantic models with correct `List[List[...]]` type hints reflecting tensor ndim
- Prometheus `Counter` and `Histogram` wired to the predict endpoint
- Inline comment shows how to swap the stub for real `onnxruntime` inference

### OpenAPIGenerator

Produces an OpenAPI 3.0 JSON spec with:

- `$ref` components for `PredictRequest` and `PredictResponse`
- Nested array schemas reflecting tensor shape and dtype
- `/predict`, `/health`, `/ready`, `/metrics` paths

### DockerfileGenerator

Multi-stage Dockerfile:

- **Stage 1 (builder):** installs Python dependencies into `/install`
- **Stage 2 (runtime):** minimal `python:3.12-slim`, non-root `nimuser`, `HEALTHCHECK`

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

---

## License

MIT
