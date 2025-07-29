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
    config = ModelConfig(
        name=name,
        max_batch_size=max_batch_size,
        dynamic_batching=dynamic_batching
    )
    
    nimifier = Nimifier(config)
    click.echo(f"Creating NIM service '{name}' from {model_path}")
    # Implementation placeholder


@main.command()
def version():
    """Show version information."""
    from . import __version__
    click.echo(f"Nimify Anything {__version__}")


if __name__ == "__main__":
    main()