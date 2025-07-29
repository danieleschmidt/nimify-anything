# Supply Chain Security

This document outlines the supply chain security practices and procedures for the Nimify project, including Software Bill of Materials (SBOM) generation, vulnerability management, and compliance frameworks.

## Overview

Supply chain security is critical for AI/ML projects due to the complex dependency tree of machine learning frameworks, GPU libraries, and inference engines. This project implements comprehensive supply chain security measures to ensure transparency, traceability, and security of all components.

## Software Bill of Materials (SBOM)

### What is an SBOM?

A Software Bill of Materials (SBOM) is a formal record containing the details and supply chain relationships of various components used in building software. It provides transparency into the software supply chain and enables better security and compliance management.

### SBOM Standards Compliance

Our project generates SBOMs in multiple standard formats:

- **SPDX 2.3**: Industry-standard format maintained by the Linux Foundation
- **CycloneDX 1.4**: OWASP standard optimized for security use cases
- **SYFT Native**: Anchore's comprehensive format with enhanced metadata

### SBOM Generation

#### Automated Generation

SBOMs are automatically generated using our comprehensive script:

```bash
# Generate complete SBOM with all formats
./scripts/generate-sbom.sh

# Generate specific components
./scripts/generate-sbom.sh python      # Python dependencies only
./scripts/generate-sbom.sh container   # Container image SBOM
./scripts/generate-sbom.sh scan        # Vulnerability scan only
```

#### Manual Generation

For development and testing purposes:

```bash
# Install required tools
curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin
pip install cyclonedx-bom

# Generate Python dependencies SBOM
cyclonedx-bom --format json --output sbom-python.json

# Generate comprehensive SBOM
syft . -o spdx-json=sbom-comprehensive.spdx.json
syft . -o cyclonedx-json=sbom-comprehensive.cyclonedx.json
```

#### CI/CD Integration

SBOM generation is integrated into our CI/CD pipeline:

```yaml
# GitHub Actions workflow step
- name: Generate SBOM
  run: |
    ./scripts/generate-sbom.sh
    # Upload SBOM artifacts
    cp sbom/reports/*-latest.* ./artifacts/
```

### SBOM Contents

Our SBOMs include comprehensive information about:

#### Core Components
- **Package Name and Version**: Exact versions of all dependencies
- **License Information**: SPDX license identifiers
- **Copyright Information**: Copyright holders and years
- **Package URLs (PURLs)**: Standardized package identifiers
- **Cryptographic Hashes**: SHA-256 checksums for verification

#### ML/AI Specific Components
- **ONNX Runtime**: Version, GPU optimization flags, execution providers
- **TensorRT**: Version, CUDA compatibility, optimization levels
- **Triton Server**: Client version, protocol support, model repository configuration
- **NVIDIA Libraries**: CUDA version, cuDNN version, driver compatibility

#### Build Environment
- **Base Container Images**: Full image digests and provenance
- **Build Tools**: Compiler versions, build flags, optimization settings
- **Runtime Environment**: Python version, system libraries, GPU drivers

## Vulnerability Management

### Automated Scanning

We employ multiple layers of vulnerability scanning:

#### Source Code Scanning
- **Bandit**: Python security linter for common security issues
- **Safety**: Known security vulnerabilities in Python packages
- **Semgrep**: Static analysis for security patterns and bugs

#### Dependency Scanning
- **Grype**: Comprehensive vulnerability database scanning
- **OSV Scanner**: Google's Open Source Vulnerabilities database
- **Trivy**: Multi-layer container and filesystem scanning

#### Container Scanning
- **Trivy**: Container image vulnerability scanning
- **Grype**: SBOM-based vulnerability assessment
- **Docker Scout**: Docker's security analysis platform

### Vulnerability Response Process

#### Critical Vulnerabilities (CVSS 9.0-10.0)
1. **Immediate Response** (within 24 hours)
   - Assess impact on production systems
   - Implement emergency patches or workarounds
   - Notify security team and stakeholders

2. **Remediation** (within 72 hours)
   - Update affected components
   - Test functionality and performance
   - Deploy fixes to all environments

#### High Vulnerabilities (CVSS 7.0-8.9)
1. **Response** (within 1 week)
   - Evaluate risk and business impact
   - Plan remediation strategy
   - Prioritize based on exploitability

2. **Remediation** (within 2 weeks)
   - Update components during regular maintenance
   - Implement additional mitigations if needed

#### Medium/Low Vulnerabilities (CVSS < 7.0)
1. **Monthly Review Cycle**
   - Bulk assessment of medium/low vulnerabilities
   - Include in regular dependency updates
   - Document acceptance of residual risk if applicable

### Vulnerability Reporting

#### Internal Reporting
- **Daily Scans**: Automated vulnerability scanning in CI/CD
- **Weekly Reports**: Summary of new vulnerabilities and remediation status
- **Monthly Reviews**: Comprehensive security posture assessment

#### External Reporting
- **Security Advisories**: Public disclosure of vulnerabilities affecting users
- **SBOM Publication**: Regular publication of updated SBOMs
- **Compliance Reports**: Regulatory and framework compliance status

## Supply Chain Security Frameworks

### SLSA (Supply-chain Levels for Software Artifacts)

Our project implements SLSA framework principles:

#### SLSA Level 1: Documentation
- ✅ **Build Process**: Documented and version-controlled build scripts
- ✅ **Provenance**: Basic build provenance information
- ✅ **SBOM**: Comprehensive software bill of materials

#### SLSA Level 2: Hosted Build Service
- ✅ **Version Control**: All source code in version control
- ✅ **Build Service**: GitHub Actions for reproducible builds
- ✅ **Build Integrity**: Protected build environments

#### SLSA Level 3: Source and Build Integrity
- 🔄 **Source Verification**: Signed commits (in progress)
- ⏳ **Isolated Builds**: Dedicated build environments (planned)
- ⏳ **Provenance Signing**: Cryptographic signing of provenance

### NIST SSDF (Secure Software Development Framework)

Implementation of NIST SSDF practices:

#### Prepare (PO)
- **PO.1.1**: Define security requirements for ML/AI components
- **PO.1.2**: Risk assessment for AI model serving infrastructure
- **PO.2.1**: Threat modeling for inference pipelines

#### Protect (PS)
- **PS.1.1**: Secure development environment configuration
- **PS.2.1**: Automated security scanning in development workflow
- **PS.3.1**: Secure coding standards for AI/ML applications

#### Produce (PW)
- **PW.1.1**: Version control for all artifacts including models
- **PW.2.1**: Automated build process with security checks
- **PW.4.1**: Comprehensive testing including security testing

#### Respond (RV)
- **RV.1.1**: Vulnerability response process
- **RV.1.2**: Security incident response procedures
- **RV.2.1**: Post-incident analysis and improvement

## Compliance and Attestations

### Regulatory Compliance

#### Industry Standards
- **ISO 27001**: Information security management
- **SOC 2 Type II**: Security, availability, and confidentiality
- **NIST Cybersecurity Framework**: Risk-based security approach

#### AI/ML Specific
- **NIST AI Risk Management Framework**: AI system risk assessment
- **EU AI Act**: Compliance for high-risk AI systems (where applicable)
- **FedRAMP**: Federal cloud security requirements (if applicable)

### SBOM Attestations

#### Digital Signatures
```bash
# Sign SBOM with GPG
gpg --detach-sign --armor sbom-comprehensive.spdx.json

# Verify signature
gpg --verify sbom-comprehensive.spdx.json.asc sbom-comprehensive.spdx.json
```

#### In-toto Attestations
```json
{
  "_type": "link",
  "name": "sbom-generation",
  "command": ["./scripts/generate-sbom.sh"],
  "materials": {
    "pyproject.toml": {"sha256": "..."},
    "src/": {"sha256": "..."}
  },
  "products": {
    "sbom/reports/comprehensive-latest.spdx.json": {"sha256": "..."}
  }
}
```

## Best Practices

### Development Practices

#### Package Management
- **Pin Exact Versions**: Critical ML libraries pinned to specific versions
- **Private PyPI**: Internal package index for proprietary components
- **License Compliance**: Automated license compatibility checking

#### Build Security
- **Reproducible Builds**: Deterministic build processes
- **Build Isolation**: Containerized build environments
- **Artifact Signing**: Digital signatures for all release artifacts

#### Dependency Management
- **Minimal Dependencies**: Reduce attack surface by minimizing dependencies
- **Regular Updates**: Automated dependency updates with security focus
- **Vulnerability Monitoring**: Continuous monitoring of dependency vulnerabilities

### Operational Practices

#### Runtime Security
- **Container Scanning**: Regular scanning of running containers
- **Network Segmentation**: Isolated inference service networks
- **Access Controls**: Principle of least privilege for service accounts

#### Monitoring and Alerting
- **Security Metrics**: Key performance indicators for security posture
- **Anomaly Detection**: Behavioral analysis for supply chain attacks
- **Incident Response**: Automated response to security events

## Tools and Integrations

### SBOM Generation Tools
- **Syft**: Comprehensive SBOM generation from various sources
- **CycloneDX Python**: Python-specific SBOM generation
- **SPDX Tools**: SPDX format validation and manipulation

### Vulnerability Scanning Tools
- **Grype**: Fast vulnerability scanning with comprehensive database
- **Trivy**: Multi-format vulnerability and misconfiguration scanning
- **OSV Scanner**: Google's open source vulnerability database

### Supply Chain Security Platforms
- **GitHub Security**: Dependabot, security advisories, code scanning
- **FOSSA**: License compliance and vulnerability management
- **Snyk**: Developer-focused security scanning and monitoring

## Getting Started

### For Developers

1. **Install Tools**:
   ```bash
   # Install SBOM generation tools
   curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh
   pip install cyclonedx-bom
   ```

2. **Generate Local SBOM**:
   ```bash
   ./scripts/generate-sbom.sh python
   ```

3. **Review Results**:
   ```bash
   cat sbom/reports/supply-chain-latest.md
   ```

### For Security Teams

1. **Review Vulnerability Scan Results**:
   ```bash
   cat sbom/vulnerabilities/vulnerabilities-latest.txt
   ```

2. **Analyze Compliance Status**:
   ```bash
   cat sbom/reports/supply-chain-latest.md | grep -A 10 "Compliance Status"
   ```

3. **Set Up Monitoring**:
   - Configure alerts for new vulnerabilities
   - Set up automated SBOM generation in CI/CD
   - Implement security metrics dashboards

## Support and Resources

### Documentation
- [NIST SSDF](https://csrc.nist.gov/Projects/ssdf)
- [SLSA Framework](https://slsa.dev/)
- [SPDX Specification](https://spdx.github.io/spdx-spec/)
- [CycloneDX Specification](https://cyclonedx.org/specification/overview/)

### Tools Documentation
- [Syft Documentation](https://github.com/anchore/syft)
- [Grype Documentation](https://github.com/anchore/grype)
- [CycloneDX Python Library](https://github.com/CycloneDX/cyclonedx-python-lib)

### Contact
- **Security Team**: security@nimify.dev
- **Supply Chain Issues**: supply-chain@nimify.dev
- **General Questions**: Create an issue in this repository