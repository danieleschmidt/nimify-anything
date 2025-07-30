FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Install additional system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
        vim \
        htop \
        tree \
        jq \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Docker CLI
RUN curl -fsSL https://get.docker.com | sh

# Install kubectl and helm
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" \
    && install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl \
    && rm kubectl

RUN curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 \
    && chmod 700 get_helm.sh \
    && ./get_helm.sh \
    && rm get_helm.sh

# Set up Python environment
RUN pip install --upgrade pip setuptools wheel

# Configure git (will be overridden by user config)
RUN git config --global init.defaultBranch main \
    && git config --global pull.rebase false

# Create workspace directory
WORKDIR /workspace

# Copy and install project dependencies
COPY requirements.txt* ./
COPY pyproject.toml ./
RUN pip install -e .[dev] || true

USER vscode