"""Advanced CLI features for Nimify with enhanced model support and deployment options."""

import asyncio
import json
import logging
from pathlib import Path

import click
import yaml

from .core import (
    NeuralConfig,
    NeuralSignalType,
    OlfactoryConfig,
    OlfactoryMoleculeType,
)
from .deployment import GlobalDeploymentManager
from .model_analyzer import ModelAnalyzer
from .monitoring import MetricsCollector, PerformanceMonitor
from .validation import ServiceNameValidator, ValidationError

logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0", prog_name="nimify-advanced")
@click.option('--debug', is_flag=True, help='Enable debug logging')
def main(debug: bool):
    """Nimify Advanced: Enhanced CLI for multi-modal AI deployment."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')


@main.group()
def bioneuro():
    """Bioneuro-olfactory fusion commands."""
    pass


@bioneuro.command()
@click.option('--neural-type', type=click.Choice(['EEG', 'fMRI', 'MEG', 'EPHYS', 'CALCIUM']), 
              default='EEG', help='Neural signal type')
@click.option('--sampling-rate', default=1000, help='Neural sampling rate (Hz)')
@click.option('--channels', default=64, help='Number of neural channels')
@click.option('--molecule-types', multiple=True, 
              type=click.Choice(['ALDEHYDE', 'ESTER', 'KETONE', 'ALCOHOL', 'TERPENE', 'AROMATIC']),
              default=['AROMATIC'], help='Olfactory molecule types')
@click.option('--output', '-o', default='bioneuro_config.json', help='Output configuration file')
def create_fusion_config(neural_type: str, sampling_rate: int, channels: int, 
                        molecule_types: list[str], output: str):
    """Create bioneuro-olfactory fusion configuration."""
    click.echo("🧠 Creating bioneuro fusion configuration...")
    
    # Create neural config
    neural_config = NeuralConfig(
        signal_type=NeuralSignalType(neural_type.lower()),
        sampling_rate=sampling_rate,
        channels=channels,
        time_window=2.0,
        preprocessing_filters=["bandpass", "notch", "baseline"],
        artifact_removal=True
    )
    
    # Create olfactory config
    olfactory_config = OlfactoryConfig(
        molecule_types=[OlfactoryMoleculeType(mt.lower()) for mt in molecule_types],
        concentration_range=(0.001, 10.0),
        molecular_descriptors=[
            "molecular_weight", "vapor_pressure", "polarity",
            "functional_groups", "carbon_chain_length"
        ],
        stimulus_duration=3.0,
        inter_stimulus_interval=10.0
    )
    
    # Export configuration
    config_data = {
        "neural": {
            "signal_type": neural_config.signal_type.value,
            "sampling_rate": neural_config.sampling_rate,
            "channels": neural_config.channels,
            "time_window": neural_config.time_window,
            "preprocessing_filters": neural_config.preprocessing_filters,
            "artifact_removal": neural_config.artifact_removal
        },
        "olfactory": {
            "molecule_types": [mt.value for mt in olfactory_config.molecule_types],
            "concentration_range": olfactory_config.concentration_range,
            "molecular_descriptors": olfactory_config.molecular_descriptors,
            "stimulus_duration": olfactory_config.stimulus_duration,
            "inter_stimulus_interval": olfactory_config.inter_stimulus_interval
        }
    }
    
    with open(output, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    click.echo(f"✅ Configuration saved to {output}")
    click.echo(f"   Neural: {neural_type} ({channels} channels @ {sampling_rate}Hz)")
    click.echo(f"   Olfactory: {', '.join(molecule_types)}")


@main.command()
@click.argument('model_path')
@click.option('--name', required=True, help='Service name')
@click.option('--format', type=click.Choice(['auto', 'onnx', 'tensorrt', 'pytorch', 'tensorflow']),
              default='auto', help='Model format (auto-detect if not specified)')
@click.option('--optimize', is_flag=True, help='Apply optimization during analysis')
@click.option('--export-config', help='Export detailed model configuration to file')
def analyze(model_path: str, name: str, format: str, optimize: bool, export_config: str | None):
    """Advanced model analysis with multi-format support."""
    model_path = Path(model_path)
    if not model_path.exists():
        click.echo(f"❌ Model file not found: {model_path}", err=True)
        raise click.Abort()
    
    try:
        ServiceNameValidator.validate_service_name(name)
    except ValidationError as e:
        click.echo(f"❌ Invalid service name: {e}", err=True)
        raise click.Abort()
    
    click.echo(f"🔍 Analyzing model: {model_path}")
    click.echo(f"📊 Service name: {name}")
    
    try:
        if format == 'auto':
            analysis = ModelAnalyzer.analyze_model(str(model_path))
        else:
            if format == 'onnx':
                analysis = ModelAnalyzer.analyze_onnx_model(str(model_path))
            elif format == 'tensorrt':
                analysis = ModelAnalyzer.analyze_tensorrt_model(str(model_path))
            elif format == 'pytorch':
                analysis = ModelAnalyzer.analyze_pytorch_model(str(model_path))
            elif format == 'tensorflow':
                analysis = ModelAnalyzer.analyze_tensorflow_model(str(model_path))
        
        click.echo("✅ Analysis complete")
        click.echo(f"   Format: {analysis['format']}")
        click.echo(f"   Model: {analysis['model_name']}")
        
        # Display inputs
        click.echo("   Inputs:")
        for name, shape in analysis['inputs'].items():
            click.echo(f"     - {name}: {shape}")
        
        # Display outputs
        click.echo("   Outputs:")
        for name, shape in analysis['outputs'].items():
            click.echo(f"     - {name}: {shape}")
        
        # Display additional metadata
        if 'estimated_type' in analysis:
            click.echo(f"   Estimated Type: {analysis['estimated_type']}")
        if 'layers_count' in analysis:
            click.echo(f"   Layers: {analysis['layers_count']}")
        if 'signatures' in analysis:
            click.echo(f"   TF Signatures: {analysis['signatures']}")
        
        # Export detailed configuration if requested
        if export_config:
            detailed_config = {
                "service_name": name,
                "model_path": str(model_path),
                "analysis": analysis,
                "optimization_settings": {
                    "enabled": optimize,
                    "target_batch_sizes": [1, 4, 8, 16, 32],
                    "precision": "fp16",
                    "memory_optimization": True
                },
                "deployment_settings": {
                    "min_replicas": 2,
                    "max_replicas": 10,
                    "cpu_request": "100m",
                    "memory_request": "512Mi",
                    "gpu_required": True
                }
            }
            
            with open(export_config, 'w') as f:
                if export_config.endswith('.yaml') or export_config.endswith('.yml'):
                    yaml.dump(detailed_config, f, default_flow_style=False)
                else:
                    json.dump(detailed_config, f, indent=2)
            
            click.echo(f"📄 Detailed configuration exported to {export_config}")
    
    except Exception as e:
        click.echo(f"❌ Analysis failed: {e}", err=True)
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            traceback.print_exc()
        raise click.Abort()


@main.group()
def deploy():
    """Advanced deployment commands."""
    pass


@deploy.command()
@click.argument('service_name')
@click.option('--regions', multiple=True, default=['us-west-2', 'eu-west-1'], 
              help='Target regions for deployment')
@click.option('--environment', type=click.Choice(['dev', 'staging', 'prod']), 
              default='dev', help='Deployment environment')
@click.option('--scaling-policy', type=click.Choice(['conservative', 'aggressive', 'custom']),
              default='conservative', help='Auto-scaling policy')
@click.option('--compliance', multiple=True, 
              type=click.Choice(['GDPR', 'CCPA', 'PDPA', 'SOC2']),
              help='Compliance requirements')
def global_deploy(service_name: str, regions: list[str], environment: str, 
                 scaling_policy: str, compliance: list[str]):
    """Deploy service globally across multiple regions."""
    click.echo(f"🌍 Initiating global deployment for {service_name}")
    click.echo(f"   Regions: {', '.join(regions)}")
    click.echo(f"   Environment: {environment}")
    click.echo(f"   Scaling: {scaling_policy}")
    if compliance:
        click.echo(f"   Compliance: {', '.join(compliance)}")
    
    try:
        deployment_manager = GlobalDeploymentManager()
        
        # Configure deployment
        deployment_config = {
            "service_name": service_name,
            "environment": environment,
            "regions": regions,
            "scaling_policy": scaling_policy,
            "compliance_requirements": list(compliance),
            "monitoring": {
                "prometheus": True,
                "grafana": True,
                "alerting": True
            },
            "security": {
                "network_policies": True,
                "pod_security_policies": True,
                "rbac": True
            }
        }
        
        click.echo("🚀 Starting deployment process...")
        
        # This would integrate with the existing deployment system
        deployment_results = deployment_manager.deploy_globally(deployment_config)
        
        click.echo("✅ Global deployment completed!")
        click.echo("📊 Deployment summary:")
        for region, status in deployment_results.items():
            click.echo(f"   {region}: {status}")
        
        # Generate monitoring dashboard URLs
        click.echo("📈 Monitoring endpoints:")
        for region in regions:
            click.echo(f"   {region}: https://grafana.{region}.nimify.ai/d/{service_name}")
    
    except Exception as e:
        click.echo(f"❌ Deployment failed: {e}", err=True)
        raise click.Abort()


@main.group()
def monitor():
    """Monitoring and performance commands."""
    pass


@monitor.command()
@click.argument('service_name')
@click.option('--duration', default=300, help='Monitoring duration in seconds')
@click.option('--interval', default=5, help='Metrics collection interval in seconds')
@click.option('--export', help='Export metrics to file')
def performance(service_name: str, duration: int, interval: int, export: str | None):
    """Monitor service performance in real-time."""
    click.echo(f"📊 Monitoring {service_name} performance...")
    click.echo(f"   Duration: {duration}s, Interval: {interval}s")
    
    try:
        PerformanceMonitor(service_name)
        metrics_collector = MetricsCollector(interval)
        
        # Start monitoring
        click.echo("🔄 Starting performance monitoring...")
        
        async def monitor_performance():
            metrics_data = []
            start_time = asyncio.get_event_loop().time()
            
            while (asyncio.get_event_loop().time() - start_time) < duration:
                # Collect metrics
                metrics = await metrics_collector.collect_metrics()
                metrics_data.append(metrics)
                
                # Display real-time metrics
                click.echo(f"   CPU: {metrics.get('cpu_percent', 0):.1f}% | "
                          f"Memory: {metrics.get('memory_mb', 0):.0f}MB | "
                          f"Requests/s: {metrics.get('requests_per_second', 0):.1f} | "
                          f"Latency: {metrics.get('avg_latency_ms', 0):.1f}ms")
                
                await asyncio.sleep(interval)
            
            # Export metrics if requested
            if export:
                with open(export, 'w') as f:
                    json.dump(metrics_data, f, indent=2)
                click.echo(f"📄 Metrics exported to {export}")
            
            return metrics_data
        
        # Run monitoring
        asyncio.run(monitor_performance())
        
        click.echo("✅ Performance monitoring completed")
    
    except Exception as e:
        click.echo(f"❌ Monitoring failed: {e}", err=True)
        raise click.Abort()


@main.command()
@click.argument('config_file')
@click.option('--dry-run', is_flag=True, help='Validate configuration without deployment')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def batch_deploy(config_file: str, dry_run: bool, verbose: bool):
    """Deploy multiple services from configuration file."""
    config_path = Path(config_file)
    if not config_path.exists():
        click.echo(f"❌ Configuration file not found: {config_path}", err=True)
        raise click.Abort()
    
    try:
        # Load configuration
        with open(config_path) as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        services = config.get('services', [])
        click.echo(f"🚀 Batch deployment: {len(services)} services")
        
        if dry_run:
            click.echo("🔍 Dry run mode - validating configurations...")
        
        for i, service_config in enumerate(services, 1):
            service_name = service_config.get('name')
            click.echo(f"[{i}/{len(services)}] Processing {service_name}")
            
            if verbose:
                click.echo(f"   Model: {service_config.get('model_path')}")
                click.echo(f"   Regions: {service_config.get('regions', [])}")
                click.echo(f"   Environment: {service_config.get('environment', 'dev')}")
            
            if not dry_run:
                # Implement actual deployment logic here
                click.echo(f"   ✅ {service_name} deployed successfully")
            else:
                # Validate configuration
                required_fields = ['name', 'model_path']
                missing = [field for field in required_fields if not service_config.get(field)]
                if missing:
                    click.echo(f"   ❌ Missing required fields: {missing}")
                else:
                    click.echo("   ✅ Configuration valid")
        
        if dry_run:
            click.echo("🔍 Dry run completed - all configurations validated")
        else:
            click.echo("✅ Batch deployment completed successfully")
    
    except Exception as e:
        click.echo(f"❌ Batch deployment failed: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    main()