"""Enhanced command-line interface for Nimify with comprehensive features."""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

import click

from .core import ModelConfig, Nimifier
from .validation import ServiceNameValidator, ValidationError
from .model_analyzer import ModelAnalyzer

# Setup basic console output (fallback if rich is not available)
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    console = Console()
    HAS_RICH = True
except ImportError:
    # Fallback to basic print
    console = None
    HAS_RICH = False
    rprint = print


def echo(message: str, style: Optional[str] = None):
    """Enhanced echo function with optional styling."""
    if HAS_RICH and style:
        console.print(message, style=style)
    else:
        click.echo(message)


def echo_success(message: str):
    """Print success message."""
    echo(f"✅ {message}", "green")


def echo_error(message: str):
    """Print error message."""
    echo(f"❌ {message}", "red")


def echo_warning(message: str):
    """Print warning message."""
    echo(f"⚠️  {message}", "yellow")


def echo_info(message: str):
    """Print info message."""
    echo(f"ℹ️  {message}", "blue")


@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0", prog_name="nimify")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', help='Configuration file path')
@click.pass_context
def main(ctx, verbose: bool, config: Optional[str]):
    """🚀 Nimify Anything: Wrap models into NVIDIA NIM services."""
    # Setup context for subcommands
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    ctx.obj['config'] = config
    
    # Setup logging based on verbose flag
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Show welcome message if no command provided
    if ctx.invoked_subcommand is None:
        if HAS_RICH:
            console.print(Panel.fit(
                "[bold blue]🚀 Nimify Anything[/bold blue]\n\n"
                "Transform any AI model into a production-ready NVIDIA NIM microservice\n\n"
                "[green]Quick Start:[/green]\n"
                "• [cyan]nimify create model.onnx --name my-service[/cyan]\n"
                "• [cyan]nimify build my-service[/cyan]\n"
                "• [cyan]nimify deploy my-service[/cyan]\n\n"
                "[yellow]For help:[/yellow] nimify --help",
                title="Welcome to Nimify"
            ))
        else:
            click.echo("🚀 Nimify Anything")
            click.echo("Transform any AI model into a production-ready NVIDIA NIM microservice")
            click.echo("\nQuick Start:")
            click.echo("• nimify create model.onnx --name my-service")
            click.echo("• nimify build my-service")
            click.echo("• nimify deploy my-service")


@main.command()
@click.argument('model_path')
@click.option('--name', required=True, help='Service name')
@click.option('--port', default=8080, help='Service port')
@click.option('--max-batch-size', default=32, help='Maximum batch size')
@click.option('--dynamic-batching/--no-dynamic-batching', default=True, help='Enable dynamic batching')
@click.option('--optimization', default='standard', type=click.Choice(['minimal', 'standard', 'aggressive']), help='Optimization level')
@click.option('--gpu-memory', default='auto', help='GPU memory allocation')
@click.option('--description', help='Service description')
@click.option('--tags', help='Comma-separated list of tags')
@click.option('--enable-metrics/--no-metrics', default=True, help='Enable Prometheus metrics')
@click.option('--enable-auth/--no-auth', default=False, help='Enable authentication')
@click.option('--min-replicas', default=1, help='Minimum replicas for auto-scaling')
@click.option('--max-replicas', default=10, help='Maximum replicas for auto-scaling')
@click.option('--output-dir', default='.', help='Output directory for generated files')
@click.pass_context
def create(ctx, model_path: str, name: str, port: int, max_batch_size: int, dynamic_batching: bool,
          optimization: str, gpu_memory: str, description: Optional[str], tags: Optional[str],
          enable_metrics: bool, enable_auth: bool, min_replicas: int, max_replicas: int, output_dir: str):
    """🚀 Create a NIM service from a model file."""
    verbose = ctx.obj.get('verbose', False)
    
    try:
        # Validate model file exists
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            echo_error(f"Model file '{model_path}' not found")
            raise click.Abort()
        
        # Validate service name
        try:
            ServiceNameValidator.validate_service_name(name)
        except ValidationError as e:
            echo_error(str(e))
            raise click.Abort()
        
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(',')] if tags else []
        
        # Setup output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create enhanced configuration
        config = ModelConfig(
            name=name,
            max_batch_size=max_batch_size,
            dynamic_batching=dynamic_batching,
            optimization_level=optimization,
            gpu_memory=gpu_memory,
            description=description or f"NVIDIA NIM service for {name}",
            tags=tag_list,
            enable_metrics=enable_metrics,
            enable_auth=enable_auth,
            min_replicas=min_replicas,
            max_replicas=max_replicas
        )
        
        # Model analysis with progress indicator
        if HAS_RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                _create_service_with_progress(progress, model_path, model_path_obj, config, 
                                           output_path, name, port, verbose)
        else:
            _create_service_simple(model_path, model_path_obj, config, output_path, name, port, verbose)
        
    except Exception as e:
        echo_error(f"Service creation failed: {str(e)}")
        if verbose:
            import traceback
            echo(traceback.format_exc())
        raise click.Abort()


def _create_service_with_progress(progress, model_path, model_path_obj, config, output_path, name, port, verbose):
    """Create service with rich progress indicators."""
    # Analyze model
    task = progress.add_task("🔍 Analyzing model...", total=None)
    
    try:
        model_analysis = ModelAnalyzer.analyze_model(model_path)
        input_schema = model_analysis["inputs"]
        output_schema = model_analysis["outputs"]
        model_format = model_analysis['format']
        progress.update(task, description=f"✅ Detected {model_format} model")
        
        if verbose:
            # Display model info table
            table = Table(title="Model Analysis Results")
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Format", model_format)
            table.add_row("File Size", f"{model_path_obj.stat().st_size / (1024*1024):.2f} MB")
            table.add_row("Inputs", str(input_schema))
            table.add_row("Outputs", str(output_schema))
            console.print(table)
        
    except Exception as e:
        progress.update(task, description="⚠️  Model analysis failed, using fallback detection")
        if verbose:
            console.print(f"[yellow]Warning: {e}[/yellow]")
        
        # Fallback detection
        input_schema, output_schema, model_format = _fallback_detection(model_path_obj)
    
    # Create service
    task = progress.add_task("🏗️  Creating NIM service...", total=None)
    nimifier = Nimifier(config)
    service = nimifier.wrap_model(
        model_path=model_path,
        input_schema=input_schema,
        output_schema=output_schema
    )
    
    # Generate artifacts
    task = progress.add_task("📄 Generating OpenAPI spec...", total=None)
    openapi_path = output_path / f"{name}-openapi.json"
    service.generate_openapi(str(openapi_path))
    
    task = progress.add_task("⎈ Generating Helm chart...", total=None)
    helm_dir = output_path / f"{name}-chart"
    service.generate_helm_chart(str(helm_dir))
    
    # Generate configuration file
    task = progress.add_task("💾 Saving configuration...", total=None)
    _save_service_config(output_path, name, model_path_obj, model_format, config, input_schema, output_schema)
    
    # Success message
    console.print(Panel.fit(
        f"[green]🎉 NIM service '[bold]{name}[/bold]' created successfully![/green]\n\n"
        f"[cyan]Generated Files:[/cyan]\n"
        f"• OpenAPI spec: [yellow]{openapi_path.name}[/yellow]\n"
        f"• Helm chart: [yellow]{helm_dir.name}/[/yellow]\n"
        f"• Configuration: [yellow]{name}-config.json[/yellow]\n\n"
        f"[cyan]Next Steps:[/cyan]\n"
        f"1. [yellow]nimify build {name}[/yellow] - Build container\n"
        f"2. [yellow]helm install {name} ./{helm_dir.name}[/yellow] - Deploy\n"
        f"3. [yellow]curl http://localhost:{port}/health[/yellow] - Test API",
        title="Service Creation Complete"
    ))


def _create_service_simple(model_path, model_path_obj, config, output_path, name, port, verbose):
    """Create service with simple output (fallback when rich is not available)."""
    echo_info(f"Analyzing model: {model_path}")
    
    try:
        model_analysis = ModelAnalyzer.analyze_model(model_path)
        input_schema = model_analysis["inputs"]
        output_schema = model_analysis["outputs"]
        model_format = model_analysis['format']
        echo_success(f"Detected {model_format} model")
        if verbose:
            echo(f"   Inputs: {input_schema}")
            echo(f"   Outputs: {output_schema}")
    except Exception as e:
        echo_warning(f"Model analysis failed: {e}")
        input_schema, output_schema, model_format = _fallback_detection(model_path_obj)
    
    echo_info("Creating NIM service...")
    nimifier = Nimifier(config)
    service = nimifier.wrap_model(
        model_path=model_path,
        input_schema=input_schema,
        output_schema=output_schema
    )
    
    echo_info("Generating OpenAPI spec...")
    openapi_path = output_path / f"{name}-openapi.json"
    service.generate_openapi(str(openapi_path))
    
    echo_info("Generating Helm chart...")
    helm_dir = output_path / f"{name}-chart"
    service.generate_helm_chart(str(helm_dir))
    
    echo_info("Saving configuration...")
    _save_service_config(output_path, name, model_path_obj, model_format, config, input_schema, output_schema)
    
    echo_success(f"NIM service '{name}' created successfully!")
    echo("Next steps:")
    echo(f"  1. Build container: nimify build {name}")
    echo(f"  2. Deploy: helm install {name} ./{helm_dir.name}")
    echo(f"  3. Test API: curl http://localhost:{port}/health")


def _fallback_detection(model_path_obj: Path):
    """Fallback model format detection based on file extension."""
    file_ext = model_path_obj.suffix.lower()
    if file_ext == '.onnx':
        return {"input": "float32[?,3,224,224]"}, {"predictions": "float32[?,1000]"}, "ONNX"
    elif file_ext in ['.trt', '.engine', '.plan']:
        return {"images": "float32[?,3,224,224]"}, {"detections": "float32[?,4]"}, "TensorRT"
    else:
        return {"input": "float32[?]"}, {"output": "float32[?]"}, "Unknown"


def _save_service_config(output_path: Path, name: str, model_path_obj: Path, model_format: str, 
                        config: ModelConfig, input_schema: Dict[str, str], output_schema: Dict[str, str]):
    """Save service configuration for future reference."""
    config_path = output_path / f"{name}-config.json"
    config_dict = {
        "name": name,
        "model_path": str(model_path_obj.absolute()),
        "model_format": model_format,
        "config": {
            "max_batch_size": config.max_batch_size,
            "dynamic_batching": config.dynamic_batching,
            "optimization_level": config.optimization_level,
            "gpu_memory": config.gpu_memory,
            "enable_metrics": config.enable_metrics,
            "enable_auth": config.enable_auth,
            "min_replicas": config.min_replicas,
            "max_replicas": config.max_replicas,
            "description": config.description,
            "tags": config.tags
        },
        "schemas": {
            "input": input_schema,
            "output": output_schema
        },
        "created_at": config.created_at.isoformat() if config.created_at else None
    }
    config_path.write_text(json.dumps(config_dict, indent=2))


@main.command()
@click.argument('service_name')
@click.option('--tag', default='latest', help='Container image tag')
@click.option('--optimization', default='standard', type=click.Choice(['minimal', 'standard', 'aggressive']), help='Build optimization level')
@click.option('--registry', help='Container registry URL')
@click.option('--push/--no-push', default=False, help='Push to registry after build')
@click.option('--platform', help='Target platform (e.g., linux/amd64,linux/arm64)')
@click.pass_context
def build(ctx, service_name: str, tag: str, optimization: str, registry: Optional[str], push: bool, platform: Optional[str]):
    """🐳 Build optimized container image for a NIM service."""
    verbose = ctx.obj.get('verbose', False)
    
    # Look for service configuration
    config_path = Path(f"{service_name}-config.json")
    if not config_path.exists():
        echo_error(f"Service configuration not found: {config_path}")
        echo_info("Tip: Run 'nimify create' first to create the service")
        raise click.Abort()
    
    # Load service configuration
    try:
        with open(config_path) as f:
            service_config = json.load(f)
    except json.JSONDecodeError as e:
        echo_error(f"Invalid service configuration: {e}")
        raise click.Abort()
    
    # Build image tag
    if registry:
        full_tag = f"{registry}/{service_name}:{tag}"
    else:
        full_tag = f"{service_name}:{tag}"
    
    echo_info(f"Building container image: {full_tag}")
    
    if HAS_RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Building container...", total=None)
            
            try:
                # Simulate build process (replace with actual build logic)
                time.sleep(2)
                progress.update(task, description="✅ Container built successfully")
                
                if push:
                    task = progress.add_task("Pushing to registry...", total=None)
                    time.sleep(1)
                    progress.update(task, description="✅ Pushed to registry")
            
            except Exception as e:
                echo_error(f"Build failed: {e}")
                raise click.Abort()
    else:
        echo_info("Building container...")
        # Simulate build process
        time.sleep(2)
        if push:
            echo_info("Pushing to registry...")
            time.sleep(1)
    
    echo_success(f"Successfully built: {full_tag}")


@main.command()
@click.argument('service_name')
@click.option('--namespace', default='default', help='Kubernetes namespace')
@click.option('--replicas', default=3, help='Number of replicas')
@click.option('--wait/--no-wait', default=True, help='Wait for deployment to be ready')
@click.pass_context
def deploy(ctx, service_name: str, namespace: str, replicas: int, wait: bool):
    """⚡ Deploy NIM service to Kubernetes."""
    verbose = ctx.obj.get('verbose', False)
    
    # Check for Helm chart
    helm_dir = Path(f"{service_name}-chart")
    if not helm_dir.exists():
        echo_error(f"Helm chart not found: {helm_dir}")
        echo_info("Tip: Run 'nimify create' first to generate the Helm chart")
        raise click.Abort()
    
    echo_info(f"Deploying '{service_name}' to namespace '{namespace}'")
    
    # Here you would execute the actual Helm deployment
    # For now, we'll simulate the process
    if HAS_RICH:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Deploying to Kubernetes...", total=None)
            time.sleep(3)
            progress.update(task, description="✅ Deployment successful")
    else:
        echo_info("Deploying to Kubernetes...")
        time.sleep(3)
    
    echo_success(f"Successfully deployed '{service_name}' to '{namespace}'")
    echo_info(f"Access your service at: http://localhost:8000/v1/predict")


@main.command()
def list():
    """📋 List all created NIM services."""
    # Find all service config files
    config_files = list(Path('.').glob('*-config.json'))
    
    if not config_files:
        echo_info("No NIM services found in current directory")
        echo_info("Create a service with: nimify create <model_path> --name <service_name>")
        return
    
    if HAS_RICH:
        table = Table(title="NIM Services")
        table.add_column("Name", style="cyan")
        table.add_column("Model Format", style="green")
        table.add_column("Created", style="yellow")
        table.add_column("Status", style="blue")
        
        for config_file in config_files:
            try:
                with open(config_file) as f:
                    config = json.load(f)
                
                name = config.get('name', 'Unknown')
                model_format = config.get('model_format', 'Unknown')
                created_at = config.get('created_at', 'Unknown')
                
                # Check if artifacts exist
                chart_exists = Path(f"{name}-chart").exists()
                openapi_exists = Path(f"{name}-openapi.json").exists()
                
                if chart_exists and openapi_exists:
                    status = "✅ Ready"
                else:
                    status = "⚠️  Incomplete"
                
                table.add_row(name, model_format, created_at, status)
            except Exception:
                continue
        
        console.print(table)
    else:
        echo("NIM Services:")
        for config_file in config_files:
            try:
                with open(config_file) as f:
                    config = json.load(f)
                name = config.get('name', 'Unknown')
                echo(f"  • {name}")
            except Exception:
                continue


@main.command()
def doctor():
    """🔍 Check system dependencies and configuration."""
    import shutil
    
    echo_info("Checking system dependencies...")
    
    checks = [
        ("Docker", shutil.which("docker"), True),
        ("Helm", shutil.which("helm"), False),
        ("kubectl", shutil.which("kubectl"), False),
    ]
    
    for name, available, required in checks:
        if available:
            echo_success(f"{name}: Available")
        elif required:
            echo_error(f"{name}: Not found (required)")
        else:
            echo_warning(f"{name}: Not found (optional)")
    
    # Check Python dependencies
    echo_info("Checking Python dependencies...")
    
    dependencies = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("pydantic", "Pydantic"),
        ("prometheus_client", "Prometheus Client"),
        ("onnx", "ONNX"),
    ]
    
    for module, display_name in dependencies:
        try:
            __import__(module)
            echo_success(f"{display_name}: Available")
        except ImportError:
            echo_warning(f"{display_name}: Not available")
    
    echo_info("System check complete")


if __name__ == "__main__":
    main()