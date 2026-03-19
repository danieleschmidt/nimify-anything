"""nimify CLI: wrap any ONNX model into an NVIDIA NIM-style microservice."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from .inspector import ModelInspector
from .generator import NIMServiceGenerator, OpenAPIGenerator, DockerfileGenerator


@click.command()
@click.argument("model_path", type=click.Path())
@click.option(
    "--output", "-o",
    default="./service",
    show_default=True,
    help="Output directory for generated service files.",
)
@click.option(
    "--mock",
    is_flag=True,
    default=False,
    help="Use mock model metadata (for demo/testing without a real .onnx file).",
)
@click.version_option(package_name="nimify-anything", prog_name="nimify")
def main(model_path: str, output: str, mock: bool) -> None:
    """Wrap MODEL_PATH (an ONNX file) into an NVIDIA NIM-style microservice.

    Generates:
      <output>/app.py          — FastAPI application stub
      <output>/openapi.json    — OpenAPI 3.0 spec
      <output>/Dockerfile      — Optimised multi-stage Docker build
    """
    output_dir = Path(output)
    click.echo(f"🔍  Inspecting model: {model_path}")

    inspector = ModelInspector(model_path, mock=mock)
    try:
        meta = inspector.inspect()
    except RuntimeError as exc:
        click.echo(f"❌  {exc}", err=True)
        sys.exit(1)

    click.echo(f"✅  Model: {meta.name!r}")
    click.echo(f"    Opset:   {meta.opset_version}")
    click.echo(f"    Inputs:  {[f'{t.name}{t.shape_str()}:{t.dtype}' for t in meta.inputs]}")
    click.echo(f"    Outputs: {[f'{t.name}{t.shape_str()}:{t.dtype}' for t in meta.outputs]}")
    click.echo()

    # --- FastAPI app
    click.echo(f"⚙️   Generating service …")
    app_path = NIMServiceGenerator(meta).generate(output_dir)
    click.echo(f"    ✔ {app_path}")

    # --- OpenAPI spec
    spec_path = OpenAPIGenerator(meta).generate(output_dir)
    click.echo(f"    ✔ {spec_path}")

    # --- Dockerfile
    df_path = DockerfileGenerator(meta).generate(output_dir)
    click.echo(f"    ✔ {df_path}")

    click.echo()
    click.echo("🎉  Done!  Next steps:")
    click.echo(f"    cd {output_dir}")
    click.echo("    pip install fastapi uvicorn prometheus-client pydantic")
    click.echo("    uvicorn app:app --reload")
    click.echo("    # or: docker build -t my-nim . && docker run -p 8000:8000 my-nim")
