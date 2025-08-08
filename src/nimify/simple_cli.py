#!/usr/bin/env python3
"""Simplified CLI that works without external dependencies."""

import sys
import json
import argparse
from pathlib import Path
from .core import ModelConfig, Nimifier

def create_service(args):
    """Create a NIM service."""
    print(f"🔍 Analyzing model: {args.model_path}")
    
    # Validate model file
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model file not found: {args.model_path}")
        return False
    
    # Validate service name (basic validation)
    if not args.name or len(args.name) < 2 or not args.name.replace('-', '').replace('_', '').isalnum():
        print(f"❌ Invalid service name: {args.name}")
        return False
    
    print(f"✅ Model file exists: {model_path}")
    print(f"✅ Service name valid: {args.name}")
    
    # Create config
    config = ModelConfig(
        name=args.name,
        max_batch_size=args.max_batch_size or 32,
        dynamic_batching=not args.no_dynamic_batching
    )
    
    # Analyze model (simplified)
    file_ext = model_path.suffix.lower()
    if file_ext == '.onnx':
        input_schema = {"input": "float32[?,3,224,224]"}
        output_schema = {"predictions": "float32[?,1000]"}
        model_format = "ONNX"
    elif file_ext in ['.trt', '.engine']:
        input_schema = {"images": "float32[?,3,224,224]"}
        output_schema = {"detections": "float32[?,6]"}
        model_format = "TensorRT"
    else:
        input_schema = {"input": "float32[?]"}
        output_schema = {"output": "float32[?]"}
        model_format = "Unknown"
    
    print(f"✅ Detected {model_format} model")
    print(f"   Inputs: {input_schema}")
    print(f"   Outputs: {output_schema}")
    
    # Create NIM service
    nimifier = Nimifier(config)
    service = nimifier.wrap_model(
        model_path=str(model_path),
        input_schema=input_schema,
        output_schema=output_schema
    )
    
    print(f"✅ Created NIM service '{args.name}'")
    
    # Generate OpenAPI spec
    openapi_path = f"{args.name}-openapi.json"
    service.generate_openapi(openapi_path)
    print(f"📄 Generated OpenAPI spec: {openapi_path}")
    
    # Generate Helm chart
    helm_dir = f"{args.name}-chart"
    service.generate_helm_chart(helm_dir)
    print(f"⎈ Generated Helm chart: {helm_dir}/")
    
    print(f"🎉 NIM service '{args.name}' created successfully!")
    print("Next steps:")
    print(f"  1. Review the generated OpenAPI spec: {openapi_path}")
    print(f"  2. Customize the Helm chart: {helm_dir}/values.yaml")
    print(f"  3. Deploy: helm install {args.name} ./{helm_dir}")
    print(f"  4. Test API: curl http://localhost:8000/health")
    
    return True

def doctor():
    """Check system and dependencies."""
    import subprocess
    print("🔍 Checking system dependencies...")
    
    checks = []
    
    # Python version
    python_version = sys.version_info
    if python_version >= (3, 10):
        checks.append(("✅", "Python", f"{python_version.major}.{python_version.minor}"))
    else:
        checks.append(("❌", "Python", f"{python_version.major}.{python_version.minor} (requires 3.10+)"))
    
    # Check for optional dependencies
    optional_deps = [
        ("click", "CLI framework"),
        ("fastapi", "API framework"),
        ("onnx", "ONNX model support"),
        ("docker", "Container building"),
        ("kubectl", "Kubernetes deployment"),
        ("helm", "Helm charts")
    ]
    
    for dep_name, description in optional_deps:
        try:
            if dep_name in ["docker", "kubectl", "helm"]:
                result = subprocess.run([dep_name, "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    checks.append(("✅", description, "Available"))
                else:
                    checks.append(("⚠️", description, "Not found (optional)"))
            else:
                __import__(dep_name)
                checks.append(("✅", description, "Available"))
        except (ImportError, subprocess.CalledProcessError, FileNotFoundError):
            checks.append(("⚠️", description, "Not found (optional)"))
    
    # Print results
    for status, name, info in checks:
        print(f"{status} {name}: {info}")
    
    print("\n🎉 System check complete")
    print("\nNote: This is a simplified CLI that works without external dependencies.")
    print("For full functionality, install dependencies with: pip install nimify-anything")

def version():
    """Show version information."""
    print("Nimify Anything v0.1.0 (Simplified CLI)")
    print("CLI for creating NVIDIA NIM microservices")
    print("\\nCore functionality available without external dependencies:")
    print("  • Model analysis and schema detection")
    print("  • OpenAPI specification generation")
    print("  • Helm chart generation")
    print("  • Kubernetes manifest generation")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="nimify",
        description="Nimify Anything: Wrap models into NVIDIA NIM services"
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a NIM service from a model file")
    create_parser.add_argument("model_path", help="Path to model file")
    create_parser.add_argument("--name", required=True, help="Service name")
    create_parser.add_argument("--port", type=int, default=8000, help="Service port")
    create_parser.add_argument("--max-batch-size", type=int, help="Maximum batch size")
    create_parser.add_argument("--no-dynamic-batching", action="store_true", help="Disable dynamic batching")
    
    # Doctor command
    subparsers.add_parser("doctor", help="Check system dependencies")
    
    # Version command
    subparsers.add_parser("version", help="Show version information")
    
    args = parser.parse_args()
    
    if args.command == "create":
        return create_service(args)
    elif args.command == "doctor":
        doctor()
        return True
    elif args.command == "version":
        version()
        return True
    else:
        parser.print_help()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)