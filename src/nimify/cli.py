"""Command-line interface for Nimify."""

import click
from .core import Nimifier, ModelConfig


@click.group()
@click.version_option()
def main():
    """Nimify Anything: Wrap models into NVIDIA NIM services."""
    pass


@main.command()
@click.argument('model_path')
@click.option('--name', required=True, help='Service name')
@click.option('--port', default=8080, help='Service port')
@click.option('--max-batch-size', default=32, help='Maximum batch size')
@click.option('--dynamic-batching/--no-dynamic-batching', default=True)
def create(model_path: str, name: str, port: int, max_batch_size: int, dynamic_batching: bool):
    """Create a NIM service from a model file."""
    import os
    from pathlib import Path
    
    # Validate model file exists
    if not Path(model_path).exists():
        click.echo(f"Error: Model file '{model_path}' not found", err=True)
        return
    
    config = ModelConfig(
        name=name,
        max_batch_size=max_batch_size,
        dynamic_batching=dynamic_batching
    )
    
    # Determine input/output schemas based on file extension
    file_ext = Path(model_path).suffix.lower()
    if file_ext == '.onnx':
        # Default ONNX schemas
        input_schema = {"input": "float32[?,3,224,224]"}
        output_schema = {"predictions": "float32[?,1000]"}
    elif file_ext == '.trt':
        # Default TensorRT schemas  
        input_schema = {"images": "float32[?,3,224,224]"}
        output_schema = {"detections": "float32[?,4]"}
    else:
        # Generic schema
        input_schema = {"input": "float32[?]"}
        output_schema = {"output": "float32[?]"}
    
    nimifier = Nimifier(config)
    service = nimifier.wrap_model(
        model_path=model_path,
        input_schema=input_schema,
        output_schema=output_schema
    )
    
    click.echo(f"✅ Creating NIM service '{name}' from {model_path}")
    
    # Generate OpenAPI spec
    openapi_path = f"{name}-openapi.json"
    service.generate_openapi(openapi_path)
    click.echo(f"📄 Generated OpenAPI spec: {openapi_path}")
    
    # Generate Helm chart
    helm_dir = f"{name}-chart"
    service.generate_helm_chart(helm_dir)
    click.echo(f"⎈ Generated Helm chart: {helm_dir}/")
    
    click.echo(f"🎉 NIM service '{name}' created successfully!")
    click.echo(f"Next steps:")
    click.echo(f"  1. Build container: nimify build {name}")
    click.echo(f"  2. Deploy: helm install {name} ./{helm_dir}")
    click.echo(f"  3. Test API: curl http://localhost:{port}/health")


@main.command()
@click.argument('service_name')
@click.option('--tag', default='latest', help='Container image tag')
@click.option('--optimize/--no-optimize', default=True, help='Optimize container build')
def build(service_name: str, tag: str, optimize: bool):
    """Build container image for a service."""
    click.echo(f"🐳 Building container image for '{service_name}'")
    
    # Use a dummy config for the build (in real implementation, load from saved config)
    config = ModelConfig(name=service_name)
    service = NIMService(
        config=config,
        model_path=f"{service_name}.onnx",  # Default, should be loaded from saved config
        input_schema={"input": "float32[?]"},
        output_schema={"output": "float32[?]"}
    )
    
    image_tag = f"{service_name}:{tag}"
    service.build_container(image_tag)
    click.echo(f"✅ Container built: {image_tag}")


@main.command()
def version():
    """Show version information."""
    click.echo("Nimify Anything v0.1.0")
    click.echo("CLI for creating NVIDIA NIM microservices")


@main.command()
def doctor():
    """Check system dependencies and configuration."""
    import shutil
    
    click.echo("🔍 Checking system dependencies...")
    
    # Check Docker
    if shutil.which("docker"):
        click.echo("✅ Docker: Available")
    else:
        click.echo("❌ Docker: Not found")
    
    # Check Helm
    if shutil.which("helm"):
        click.echo("✅ Helm: Available") 
    else:
        click.echo("⚠️  Helm: Not found (optional for local development)")
    
    # Check Python dependencies
    try:
        import fastapi
        click.echo("✅ FastAPI: Available")
    except ImportError:
        click.echo("❌ FastAPI: Not installed")
    
    try:
        import onnx
        click.echo("✅ ONNX: Available")
    except ImportError:
        click.echo("⚠️  ONNX: Not installed (required for ONNX models)")
    
    click.echo("🎉 System check complete")


if __name__ == "__main__":
    main()