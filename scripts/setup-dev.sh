#!/bin/bash
# Development environment setup script for Nimify Anything

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running from project root
if [[ ! -f "pyproject.toml" ]]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

print_status "Setting up Nimify Anything development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if [[ $(echo "$python_version >= $required_version" | bc -l) -eq 0 ]]; then
    print_error "Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

print_success "Python version check passed: $python_version"

# Create virtual environment if it doesn't exist
if [[ ! -d ".venv" ]]; then
    print_status "Creating virtual environment..."
    python3 -m venv .venv
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
python -m pip install --upgrade pip

# Install package in development mode
print_status "Installing package in development mode..."
pip install -e .[dev]

# Install pre-commit hooks
print_status "Installing pre-commit hooks..."
pre-commit install --install-hooks

# Initialize secrets baseline if it doesn't exist
if [[ ! -f ".secrets.baseline" ]]; then
    print_status "Initializing secrets baseline..."
    detect-secrets scan --baseline .secrets.baseline
    print_success "Secrets baseline created"
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs
mkdir -p models
mkdir -p cache
mkdir -p .cache/pip

# Set up git hooks (if git repo)
if [[ -d ".git" ]]; then
    print_status "Setting up git configuration..."
    git config --local core.autocrlf false
    git config --local core.filemode false
    print_success "Git configuration updated"
fi

# Check Docker availability
if command -v docker &> /dev/null; then
    print_success "Docker is available"
    
    # Check if Docker daemon is running
    if docker info &> /dev/null; then
        print_success "Docker daemon is running"
    else
        print_warning "Docker daemon is not running. Start Docker to use container features."
    fi
else
    print_warning "Docker is not installed. Container features will not be available."
fi

# Check NVIDIA Docker support
if command -v nvidia-docker &> /dev/null || (docker info 2>/dev/null | grep -q nvidia); then
    print_success "NVIDIA Docker support detected"
else
    print_warning "NVIDIA Docker support not detected. GPU features may not work in containers."
fi

# Run initial quality checks
print_status "Running initial code quality checks..."

print_status "Running linter..."
if ruff check src tests --quiet; then
    print_success "Linting passed"
else
    print_warning "Linting issues found. Run 'ruff check src tests' for details."
fi

print_status "Running type checker..."
if mypy src --quiet; then
    print_success "Type checking passed"
else
    print_warning "Type checking issues found. Run 'mypy src' for details."
fi

print_status "Running tests..."
if pytest --quiet; then
    print_success "All tests passed"
else
    print_warning "Some tests failed. Run 'pytest -v' for details."
fi

# Display helpful information
echo ""
print_success "Development environment setup complete!"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Activate the virtual environment: ${YELLOW}source .venv/bin/activate${NC}"
echo "2. Run tests: ${YELLOW}pytest${NC}"
echo "3. Start coding! 🚀"
echo ""
echo -e "${BLUE}Useful commands:${NC}"
echo "• Run full quality check: ${YELLOW}make lint${NC}"
echo "• Run tests with coverage: ${YELLOW}make test${NC}"
echo "• Format code: ${YELLOW}make format${NC}"
echo "• Build Docker image: ${YELLOW}make build${NC}"
echo "• Start development server: ${YELLOW}docker-compose up -d${NC}"
echo ""
echo -e "${BLUE}VS Code users:${NC}"
echo "• Open workspace: ${YELLOW}code .${NC}"
echo "• Select Python interpreter: ${YELLOW}.venv/bin/python${NC}"
echo ""

# Create .env file template if it doesn't exist
if [[ ! -f ".env" ]]; then
    cat > .env << EOF
# Environment variables for development
NIMIFY_LOG_LEVEL=DEBUG
NIMIFY_MODEL_CACHE=./cache
CUDA_VISIBLE_DEVICES=0
PYTHONPATH=./src

# Uncomment and configure as needed
# NIMIFY_REGISTRY=ghcr.io/yourusername
# NIMIFY_NAMESPACE=nimify
# NIMIFY_METRICS_PORT=9090
EOF
    print_success "Created .env file template"
fi

print_success "Setup complete! Happy coding! 🎉"