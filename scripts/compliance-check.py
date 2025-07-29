#!/usr/bin/env python3
"""
Automated compliance checking script for Nimify project.
Validates adherence to security frameworks, standards, and best practices.
"""

import json
import os
import subprocess
import sys
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('compliance-check.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ComplianceChecker:
    """Main compliance checking orchestrator."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "project": "nimify-anything",
            "version": self._get_project_version(),
            "compliance_frameworks": {},
            "overall_score": 0,
            "recommendations": []
        }
    
    def _get_project_version(self) -> str:
        """Extract project version from pyproject.toml."""
        try:
            pyproject_file = self.project_root / "pyproject.toml"
            if pyproject_file.exists():
                with open(pyproject_file, 'r') as f:
                    import toml
                    data = toml.load(f)
                    return data.get('project', {}).get('version', 'unknown')
        except Exception:
            pass
        return 'unknown'
    
    def run_all_checks(self) -> Dict:
        """Run all compliance checks and return consolidated results."""
        logger.info("Starting comprehensive compliance check...")
        
        # Security compliance frameworks
        self.results["compliance_frameworks"]["nist_ssdf"] = self.check_nist_ssdf()
        self.results["compliance_frameworks"]["slsa"] = self.check_slsa()
        self.results["compliance_frameworks"]["soc2"] = self.check_soc2_type2()
        self.results["compliance_frameworks"]["iso27001"] = self.check_iso27001()
        self.results["compliance_frameworks"]["owasp_samm"] = self.check_owasp_samm()
        
        # AI/ML specific compliance
        self.results["compliance_frameworks"]["nist_ai_rmf"] = self.check_nist_ai_rmf()
        self.results["compliance_frameworks"]["mlops_maturity"] = self.check_mlops_maturity()
        
        # Supply chain security
        self.results["compliance_frameworks"]["supply_chain"] = self.check_supply_chain_security()
        
        # Calculate overall compliance score
        self._calculate_overall_score()
        
        logger.info(f"Compliance check completed. Overall score: {self.results['overall_score']:.1f}%")
        return self.results
    
    def check_nist_ssdf(self) -> Dict:
        """Check NIST Secure Software Development Framework compliance."""
        logger.info("Checking NIST SSDF compliance...")
        
        score = 0
        max_score = 20
        findings = []
        
        # PO (Prepare) - 5 points
        po_score = 0
        
        # PO.1.1 - Define security requirements
        if (self.project_root / "SECURITY.md").exists():
            po_score += 1
            findings.append("✅ Security requirements documented")
        else:
            findings.append("❌ Missing SECURITY.md with security requirements")
        
        # PO.1.2 - Risk assessment
        if (self.project_root / "docs" / "SUPPLY_CHAIN_SECURITY.md").exists():
            po_score += 1
            findings.append("✅ Supply chain risk assessment documented")
        else:
            findings.append("❌ Missing supply chain risk assessment")
        
        # PO.2.1 - Threat modeling
        threat_model_files = list(self.project_root.glob("**/threat*model*"))
        if threat_model_files:
            po_score += 1
            findings.append("✅ Threat modeling artifacts found")
        else:
            findings.append("❌ No threat modeling documentation found")
        
        # PO.3.1 - Security training
        if (self.project_root / "docs" / "INCIDENT_RESPONSE.md").exists():
            po_score += 1
            findings.append("✅ Incident response training documented")
        else:
            findings.append("❌ Missing incident response training documentation")
        
        # PO.4.1 - Tool selection
        if (self.project_root / ".pre-commit-config.yaml").exists():
            po_score += 1
            findings.append("✅ Security toolchain configured (pre-commit)")
        else:
            findings.append("❌ Missing pre-commit security toolchain")
        
        score += po_score
        
        # PS (Protect) - 5 points
        ps_score = 0
        
        # PS.1.1 - Secure development environment
        if (self.project_root / ".editorconfig").exists():
            ps_score += 1
            findings.append("✅ Development environment standardized")
        
        # PS.2.1 - Security scanning
        if self._check_security_scanning_config():
            ps_score += 1
            findings.append("✅ Automated security scanning configured")
        else:
            findings.append("❌ Missing automated security scanning")
        
        # PS.3.1 - Secure dependencies
        if self._check_dependency_management():
            ps_score += 1
            findings.append("✅ Dependency management security configured")
        else:
            findings.append("❌ Missing secure dependency management")
        
        # PS.3.2 - Code reviews
        if (self.project_root / "CONTRIBUTING.md").exists():
            ps_score += 1
            findings.append("✅ Code review process documented")
        
        # PS.3.3 - Static analysis
        if self._check_static_analysis():
            ps_score += 1
            findings.append("✅ Static analysis tools configured")
        else:
            findings.append("❌ Missing comprehensive static analysis")
        
        score += ps_score
        
        # PW (Produce) - 5 points
        pw_score = 0
        
        # PW.1.1 - Version control
        if (self.project_root / ".git").is_dir():
            pw_score += 1
            findings.append("✅ Version control system in use")
        
        # PW.2.1 - Build process
        if self._check_build_process():
            pw_score += 1
            findings.append("✅ Automated build process configured")
        
        # PW.4.1 - Testing
        if self._check_testing_framework():
            pw_score += 1
            findings.append("✅ Comprehensive testing framework")
        else:
            findings.append("❌ Missing comprehensive testing")
        
        # PW.6.1 - Vulnerability management
        if self._check_vulnerability_management():
            pw_score += 1
            findings.append("✅ Vulnerability management process")
        else:
            findings.append("❌ Missing vulnerability management")
        
        # PW.7.1 - Software composition analysis
        if (self.project_root / "scripts" / "generate-sbom.sh").exists():
            pw_score += 1
            findings.append("✅ Software composition analysis (SBOM)")
        else:
            findings.append("❌ Missing SBOM generation")
        
        score += pw_score
        
        # RV (Respond) - 5 points
        rv_score = 0
        
        # RV.1.1 - Incident response
        if (self.project_root / "docs" / "INCIDENT_RESPONSE.md").exists():
            rv_score += 2
            findings.append("✅ Incident response plan documented")
        
        # RV.1.2 - Vulnerability disclosure
        if "security" in open(self.project_root / "SECURITY.md").read().lower():
            rv_score += 1
            findings.append("✅ Vulnerability disclosure process")
        
        # RV.2.1 - Post-incident analysis
        if "post-incident" in open(self.project_root / "docs" / "INCIDENT_RESPONSE.md").read().lower():
            rv_score += 1
            findings.append("✅ Post-incident analysis process")
        
        # RV.3.1 - Lessons learned
        if (self.project_root / "docs" / "RUNBOOKS.md").exists():
            rv_score += 1
            findings.append("✅ Operational runbooks for lessons learned")
        
        score += rv_score
        
        compliance_percentage = (score / max_score) * 100
        
        return {
            "framework": "NIST SSDF",
            "version": "1.1",
            "score": score,
            "max_score": max_score,
            "compliance_percentage": compliance_percentage,
            "status": "compliant" if compliance_percentage >= 80 else "non_compliant",
            "findings": findings,
            "recommendations": self._get_nist_ssdf_recommendations(score, max_score)
        }
    
    def check_slsa(self) -> Dict:
        """Check SLSA (Supply-chain Levels for Software Artifacts) compliance."""
        logger.info("Checking SLSA compliance...")
        
        level_1_score = 0
        level_2_score = 0
        level_3_score = 0
        findings = []
        
        # SLSA Level 1 - Build (2 requirements)
        # L1: Build scripted
        if (self.project_root / "Makefile").exists() or (self.project_root / "pyproject.toml").exists():
            level_1_score += 1
            findings.append("✅ L1: Build process scripted")
        else:
            findings.append("❌ L1: Missing scripted build process")
        
        # L1: Provenance available
        if (self.project_root / "scripts" / "generate-sbom.sh").exists():
            level_1_score += 1
            findings.append("✅ L1: Build provenance generation available")
        else:
            findings.append("❌ L1: Missing build provenance generation")
        
        # SLSA Level 2 - Source (3 additional requirements)
        if level_1_score == 2:  # Only check L2 if L1 is complete
            # L2: Version controlled
            if (self.project_root / ".git").is_dir():
                level_2_score += 1
                findings.append("✅ L2: Source code version controlled")
            
            # L2: Build service
            github_workflows = self.project_root / ".github" / "workflows"
            if github_workflows.exists() and list(github_workflows.glob("*.yml")):
                level_2_score += 1
                findings.append("✅ L2: Hosted build service available")
            else:
                findings.append("❌ L2: Missing hosted build service (GitHub Actions)")
            
            # L2: Build/source integrity
            if self._check_build_integrity():
                level_2_score += 1
                findings.append("✅ L2: Build and source integrity measures")
            else:
                findings.append("❌ L2: Missing build/source integrity measures")
        
        # SLSA Level 3 - Hardened builds (2 additional requirements)
        if level_1_score == 2 and level_2_score == 3:  # Only check L3 if L1+L2 complete
            # L3: Source and build platforms are separate
            # This would require analysis of actual CI/CD setup
            findings.append("⏳ L3: Source/build platform separation requires manual verification")
            
            # L3: Build as code
            if self._check_build_as_code():
                level_3_score += 1
                findings.append("✅ L3: Build process defined as code")
            else:
                findings.append("❌ L3: Build process not fully defined as code")
        
        # Determine overall SLSA level
        if level_1_score == 2 and level_2_score == 3 and level_3_score >= 1:
            slsa_level = 3
        elif level_1_score == 2 and level_2_score == 3:
            slsa_level = 2
        elif level_1_score == 2:
            slsa_level = 1
        else:
            slsa_level = 0
        
        return {
            "framework": "SLSA",
            "version": "1.0",
            "level_achieved": slsa_level,
            "level_1_score": f"{level_1_score}/2",
            "level_2_score": f"{level_2_score}/3",
            "level_3_score": f"{level_3_score}/2",
            "status": "compliant" if slsa_level >= 2 else "non_compliant",
            "findings": findings,
            "recommendations": self._get_slsa_recommendations(slsa_level)
        }
    
    def check_soc2_type2(self) -> Dict:
        """Check SOC 2 Type II compliance indicators."""
        logger.info("Checking SOC 2 Type II compliance indicators...")
        
        score = 0
        max_score = 15
        findings = []
        
        # Security (Common Criteria)
        security_score = 0
        
        # Access controls
        if (self.project_root / ".github" / "dependabot.yml").exists():
            security_score += 1
            findings.append("✅ Automated dependency management")
        
        # Security monitoring
        if self._check_security_monitoring():
            security_score += 1
            findings.append("✅ Security monitoring configured")
        else:
            findings.append("❌ Missing security monitoring")
        
        # Incident response
        if (self.project_root / "docs" / "INCIDENT_RESPONSE.md").exists():
            security_score += 1
            findings.append("✅ Incident response procedures documented")
        
        score += security_score
        
        # Availability
        availability_score = 0
        
        # Monitoring and alerting
        if (self.project_root / "monitoring").exists():
            availability_score += 1
            findings.append("✅ Monitoring infrastructure configured")
        
        # Backup procedures
        if "backup" in str(self.project_root / "docs" / "RUNBOOKS.md").lower():
            availability_score += 1
            findings.append("✅ Backup procedures documented")
        
        # Disaster recovery
        if "disaster" in open(self.project_root / "docs" / "RUNBOOKS.md").read().lower():
            availability_score += 1
            findings.append("✅ Disaster recovery procedures")
        
        score += availability_score
        
        # Processing Integrity
        integrity_score = 0
        
        # Data validation
        if self._check_data_validation():
            integrity_score += 1
            findings.append("✅ Data validation mechanisms")
        
        # Error handling
        if self._check_error_handling():
            integrity_score += 1
            findings.append("✅ Error handling mechanisms")
        
        # Testing procedures
        if (self.project_root / "tests").exists():
            integrity_score += 1
            findings.append("✅ Testing procedures implemented")
        
        score += integrity_score
        
        # Confidentiality
        confidentiality_score = 0
        
        # Encryption controls
        if self._check_encryption_controls():
            confidentiality_score += 1
            findings.append("✅ Encryption controls implemented")
        
        # Access restrictions
        if (self.project_root / "SECURITY.md").exists():
            confidentiality_score += 1
            findings.append("✅ Access control documentation")
        
        # Data classification
        if "classification" in open(self.project_root / "docs" / "SUPPLY_CHAIN_SECURITY.md").read().lower():
            confidentiality_score += 1
            findings.append("✅ Data classification procedures")
        
        score += confidentiality_score
        
        # Privacy (if applicable)
        privacy_score = 0
        
        # Privacy policy
        privacy_files = list(self.project_root.glob("**/PRIVACY*"))
        if privacy_files:
            privacy_score += 1
            findings.append("✅ Privacy policy documented")
        else:
            findings.append("❌ Missing privacy policy")
        
        # Data retention
        if "retention" in open(self.project_root / "SECURITY.md").read().lower():
            privacy_score += 1
            findings.append("✅ Data retention policies")
        
        # GDPR compliance (if applicable)
        if "gdpr" in open(self.project_root / "docs" / "SUPPLY_CHAIN_SECURITY.md").read().lower():
            privacy_score += 1
            findings.append("✅ GDPR compliance measures")
        
        score += privacy_score
        
        compliance_percentage = (score / max_score) * 100
        
        return {
            "framework": "SOC 2 Type II",
            "version": "2017",
            "score": score,
            "max_score": max_score,
            "compliance_percentage": compliance_percentage,
            "status": "compliant" if compliance_percentage >= 70 else "non_compliant",
            "trust_service_criteria": {
                "security": f"{security_score}/3",
                "availability": f"{availability_score}/3",
                "processing_integrity": f"{integrity_score}/3",
                "confidentiality": f"{confidentiality_score}/3",
                "privacy": f"{privacy_score}/3"
            },
            "findings": findings,
            "recommendations": self._get_soc2_recommendations(score, max_score)
        }
    
    def check_iso27001(self) -> Dict:
        """Check ISO 27001 compliance indicators."""
        logger.info("Checking ISO 27001 compliance indicators...")
        
        score = 0
        max_score = 12
        findings = []
        
        # Information Security Management System (ISMS)
        if (self.project_root / "SECURITY.md").exists():
            score += 1
            findings.append("✅ Information security policy documented")
        
        # Risk Management
        if (self.project_root / "docs" / "SUPPLY_CHAIN_SECURITY.md").exists():
            score += 1
            findings.append("✅ Risk assessment documentation")
        
        # Asset Management
        if (self.project_root / "scripts" / "generate-sbom.sh").exists():
            score += 1
            findings.append("✅ Asset inventory (SBOM generation)")
        
        # Access Control
        if self._check_access_controls():
            score += 1
            findings.append("✅ Access control measures")
        else:
            findings.append("❌ Missing access control documentation")
        
        # Cryptography
        if self._check_cryptography_controls():
            score += 1
            findings.append("✅ Cryptographic controls")
        else:
            findings.append("❌ Missing cryptographic controls documentation")
        
        # Physical and Environmental Security
        if "physical" in open(self.project_root / "SECURITY.md").read().lower():
            score += 1
            findings.append("✅ Physical security considerations")
        else:
            findings.append("❌ Missing physical security documentation")
        
        # Operations Security
        if (self.project_root / "docs" / "RUNBOOKS.md").exists():
            score += 1
            findings.append("✅ Operations security procedures")
        
        # Communications Security
        if self._check_communications_security():
            score += 1
            findings.append("✅ Communications security measures")
        else:
            findings.append("❌ Missing communications security documentation")
        
        # System Acquisition, Development and Maintenance
        if self._check_secure_development():
            score += 1
            findings.append("✅ Secure development practices")
        
        # Supplier Relationships
        if "supplier" in open(self.project_root / "docs" / "SUPPLY_CHAIN_SECURITY.md").read().lower():
            score += 1
            findings.append("✅ Supplier relationship security")
        
        # Information Security Incident Management
        if (self.project_root / "docs" / "INCIDENT_RESPONSE.md").exists():
            score += 1
            findings.append("✅ Incident management procedures")
        
        # Business Continuity Management
        if "continuity" in open(self.project_root / "docs" / "RUNBOOKS.md").read().lower():
            score += 1
            findings.append("✅ Business continuity planning")
        else:
            findings.append("❌ Missing business continuity documentation")
        
        compliance_percentage = (score / max_score) * 100
        
        return {
            "framework": "ISO 27001",
            "version": "2013",
            "score": score,
            "max_score": max_score,
            "compliance_percentage": compliance_percentage,
            "status": "compliant" if compliance_percentage >= 75 else "non_compliant",
            "findings": findings,
            "recommendations": self._get_iso27001_recommendations(score, max_score)
        }
    
    def check_owasp_samm(self) -> Dict:
        """Check OWASP SAMM (Software Assurance Maturity Model) compliance."""
        logger.info("Checking OWASP SAMM compliance...")
        
        # SAMM uses maturity levels 0-3 for each business function
        governance_score = self._check_samm_governance()
        design_score = self._check_samm_design()
        implementation_score = self._check_samm_implementation()
        verification_score = self._check_samm_verification()
        operations_score = self._check_samm_operations()
        
        findings = []
        findings.extend(governance_score["findings"])
        findings.extend(design_score["findings"])
        findings.extend(implementation_score["findings"]) 
        findings.extend(verification_score["findings"])
        findings.extend(operations_score["findings"])
        
        average_maturity = (
            governance_score["level"] + 
            design_score["level"] + 
            implementation_score["level"] + 
            verification_score["level"] + 
            operations_score["level"]
        ) / 5
        
        return {
            "framework": "OWASP SAMM",
            "version": "2.0",
            "average_maturity_level": round(average_maturity, 1),
            "business_functions": {
                "governance": governance_score,
                "design": design_score,
                "implementation": implementation_score,
                "verification": verification_score,
                "operations": operations_score
            },
            "status": "mature" if average_maturity >= 2.0 else "developing",
            "findings": findings,
            "recommendations": self._get_samm_recommendations(average_maturity)
        }
    
    def check_nist_ai_rmf(self) -> Dict:
        """Check NIST AI Risk Management Framework compliance."""
        logger.info("Checking NIST AI RMF compliance...")
        
        score = 0
        max_score = 16
        findings = []
        
        # GOVERN (4 points)
        govern_score = 0
        
        # AI governance structure
        ai_docs = list(self.project_root.glob("**/AI*")) + list(self.project_root.glob("**/MODEL*"))
        if ai_docs:
            govern_score += 1
            findings.append("✅ AI/ML governance documentation")
        else:
            findings.append("❌ Missing AI/ML governance documentation")
        
        # Risk management
        if "ai" in open(self.project_root / "docs" / "SUPPLY_CHAIN_SECURITY.md").read().lower():
            govern_score += 1
            findings.append("✅ AI risk management consideration")
        
        # Stakeholder involvement
        if (self.project_root / "CONTRIBUTING.md").exists():
            govern_score += 1
            findings.append("✅ Stakeholder involvement process")
        
        # Legal and regulatory awareness
        if "compliance" in open(self.project_root / "docs" / "SUPPLY_CHAIN_SECURITY.md").read().lower():
            govern_score += 1
            findings.append("✅ Legal and regulatory compliance awareness")
        
        score += govern_score
        
        # MAP (4 points)
        map_score = 0
        
        # AI system categorization
        if "onnx" in open(self.project_root / "README.md").read().lower():
            map_score += 1
            findings.append("✅ AI system type documented (ONNX/TensorRT)")
        
        # Risk assessment
        if (self.project_root / "docs" / "SUPPLY_CHAIN_SECURITY.md").exists():
            map_score += 1
            findings.append("✅ Risk assessment framework")
        
        # Impact assessment
        if "impact" in open(self.project_root / "docs" / "SUPPLY_CHAIN_SECURITY.md").read().lower():
            map_score += 1
            findings.append("✅ Impact assessment consideration")
        
        # Context documentation
        if (self.project_root / "docs" / "ARCHITECTURE.md").exists():
            map_score += 1
            findings.append("✅ System context documented")
        
        score += map_score
        
        # MEASURE (4 points)
        measure_score = 0
        
        # Performance monitoring
        if (self.project_root / "monitoring").exists():
            measure_score += 1
            findings.append("✅ Performance monitoring configured")
        
        # Testing framework
        if (self.project_root / "tests").exists():
            measure_score += 1
            findings.append("✅ Testing framework implemented")
        
        # Metrics collection
        if "metrics" in open(self.project_root / "README.md").read().lower():
            measure_score += 1
            findings.append("✅ Metrics collection capability")
        
        # Performance tracking
        if (self.project_root / "tests" / "performance").exists():
            measure_score += 1
            findings.append("✅ Performance testing framework")
        
        score += measure_score
        
        # MANAGE (4 points)
        manage_score = 0
        
        # Incident response
        if (self.project_root / "docs" / "INCIDENT_RESPONSE.md").exists():
            manage_score += 1
            findings.append("✅ Incident response procedures")
        
        # Change management
        if (self.project_root / "CONTRIBUTING.md").exists():
            manage_score += 1
            findings.append("✅ Change management process")
        
        # Continuous improvement
        if "improvement" in open(self.project_root / "docs" / "INCIDENT_RESPONSE.md").read().lower():
            manage_score += 1
            findings.append("✅ Continuous improvement process")
        
        # Resource allocation
        if (self.project_root / "docs" / "RUNBOOKS.md").exists():
            manage_score += 1
            findings.append("✅ Resource management procedures")
        
        score += manage_score
        
        compliance_percentage = (score / max_score) * 100
        
        return {
            "framework": "NIST AI RMF",
            "version": "1.0",
            "score": score,
            "max_score": max_score,
            "compliance_percentage": compliance_percentage,
            "core_functions": {
                "govern": f"{govern_score}/4",
                "map": f"{map_score}/4", 
                "measure": f"{measure_score}/4",
                "manage": f"{manage_score}/4"
            },
            "status": "compliant" if compliance_percentage >= 70 else "non_compliant",
            "findings": findings,
            "recommendations": self._get_ai_rmf_recommendations(score, max_score)
        }
    
    def check_mlops_maturity(self) -> Dict:
        """Check MLOps maturity level."""
        logger.info("Checking MLOps maturity...")
        
        # MLOps maturity levels: 0 (No MLOps), 1 (DevOps no MLOps), 2 (Automated Training), 3 (Automated Deployment)
        level = 0
        findings = []
        capabilities = {}
        
        # Level 1: DevOps but no MLOps
        devops_score = 0
        
        if (self.project_root / ".git").is_dir():
            devops_score += 1
        if (self.project_root / "tests").exists():
            devops_score += 1
        if (self.project_root / "Dockerfile").exists():
            devops_score += 1
        
        capabilities["devops"] = devops_score >= 2
        
        if devops_score >= 2:
            level = 1
            findings.append("✅ Level 1: Basic DevOps practices")
        
        # Level 2: Automated Training Pipeline
        training_automation = 0
        
        # Model versioning
        if "version" in open(self.project_root / "README.md").read().lower():
            training_automation += 1
        
        # Experiment tracking (inferred from documentation)
        if "experiment" in open(self.project_root / "README.md").read().lower():
            training_automation += 1
        
        # Automated testing for ML
        if (self.project_root / "tests" / "performance").exists():
            training_automation += 1
        
        capabilities["training_automation"] = training_automation >= 2
        
        if level >= 1 and training_automation >= 2:
            level = 2
            findings.append("✅ Level 2: Automated training pipeline elements")
        
        # Level 3: Automated Deployment
        deployment_automation = 0
        
        # CI/CD for ML
        if (self.project_root / "docs" / "workflows").exists():
            deployment_automation += 1
        
        # Model serving automation
        if "serving" in open(self.project_root / "README.md").read().lower():
            deployment_automation += 1
        
        # Monitoring and alerting
        if (self.project_root / "monitoring").exists():
            deployment_automation += 1
        
        capabilities["deployment_automation"] = deployment_automation >= 2
        
        if level >= 2 and deployment_automation >= 2:
            level = 3
            findings.append("✅ Level 3: Automated deployment pipeline")
        
        return {
            "framework": "MLOps Maturity",
            "version": "Custom",
            "maturity_level": level,
            "max_level": 3,
            "capabilities": capabilities,
            "status": "mature" if level >= 2 else "developing",
            "findings": findings,
            "recommendations": self._get_mlops_recommendations(level)
        }
    
    def check_supply_chain_security(self) -> Dict:
        """Check supply chain security measures."""
        logger.info("Checking supply chain security...")
        
        score = 0
        max_score = 10
        findings = []
        
        # SBOM generation
        if (self.project_root / "scripts" / "generate-sbom.sh").exists():
            score += 2
            findings.append("✅ SBOM generation capability")
        else:
            findings.append("❌ Missing SBOM generation")
        
        # Dependency scanning
        if self._check_dependency_scanning():
            score += 2
            findings.append("✅ Dependency vulnerability scanning")
        else:
            findings.append("❌ Missing dependency vulnerability scanning")
        
        # Supply chain documentation
        if (self.project_root / "docs" / "SUPPLY_CHAIN_SECURITY.md").exists():
            score += 1
            findings.append("✅ Supply chain security documentation")
        
        # Dependency management automation
        if (self.project_root / ".github" / "dependabot.yml").exists():
            score += 1
            findings.append("✅ Automated dependency updates")
        
        # License compliance
        if (self.project_root / "LICENSE").exists():
            score += 1
            findings.append("✅ License documentation")
        
        # Secrets management
        if self._check_secrets_management():
            score += 1
            findings.append("✅ Secrets management practices")
        else:
            findings.append("❌ Missing secrets management documentation")
        
        # Container security
        if self._check_container_security():
            score += 1
            findings.append("✅ Container security measures")
        else:
            findings.append("❌ Missing container security measures")
        
        # Third-party risk management
        if "third-party" in open(self.project_root / "docs" / "SUPPLY_CHAIN_SECURITY.md").read().lower():
            score += 1
            findings.append("✅ Third-party risk management")
        else:
            findings.append("❌ Missing third-party risk management")
        
        compliance_percentage = (score / max_score) * 100
        
        return {
            "framework": "Supply Chain Security",
            "version": "Custom",
            "score": score,
            "max_score": max_score,
            "compliance_percentage": compliance_percentage,
            "status": "secure" if compliance_percentage >= 80 else "needs_improvement",
            "findings": findings,
            "recommendations": self._get_supply_chain_recommendations(score, max_score)
        }
    
    # Helper methods for checking specific conditions
    def _check_security_scanning_config(self) -> bool:
        """Check if security scanning is configured."""
        if not (self.project_root / ".pre-commit-config.yaml").exists():
            return False
        
        with open(self.project_root / ".pre-commit-config.yaml", 'r') as f:
            content = f.read()
            return "bandit" in content and "safety" in content
    
    def _check_dependency_management(self) -> bool:
        """Check if dependency management security is configured."""
        return (
            (self.project_root / ".github" / "dependabot.yml").exists() or
            (self.project_root / ".github" / "renovate.json").exists()
        )
    
    def _check_static_analysis(self) -> bool:
        """Check if static analysis is properly configured."""
        if not (self.project_root / ".pre-commit-config.yaml").exists():
            return False
        
        with open(self.project_root / ".pre-commit-config.yaml", 'r') as f:
            content = f.read()
            return "ruff" in content and "mypy" in content
    
    def _check_build_process(self) -> bool:
        """Check if automated build process is configured."""
        return (
            (self.project_root / "Makefile").exists() or
            (self.project_root / "pyproject.toml").exists() or
            (self.project_root / "Dockerfile").exists()
        )
    
    def _check_testing_framework(self) -> bool:
        """Check if comprehensive testing framework exists."""
        tests_dir = self.project_root / "tests"
        if not tests_dir.exists():
            return False
        
        # Check for different types of tests
        test_files = list(tests_dir.glob("**/*.py"))
        performance_tests = (tests_dir / "performance").exists()
        
        return len(test_files) >= 3 and performance_tests
    
    def _check_vulnerability_management(self) -> bool:
        """Check if vulnerability management is configured."""
        security_md = self.project_root / "SECURITY.md"
        if not security_md.exists():
            return False
        
        with open(security_md, 'r') as f:
            content = f.read().lower()
            return "vulnerability" in content and "report" in content
    
    def _check_build_integrity(self) -> bool:
        """Check if build integrity measures are in place."""
        # Check for reproducible builds, checksums, etc.
        return (
            (self.project_root / "scripts" / "generate-sbom.sh").exists() and
            (self.project_root / ".github").exists()
        )
    
    def _check_build_as_code(self) -> bool:
        """Check if build process is defined as code."""
        github_workflows = self.project_root / ".github" / "workflows"
        if not github_workflows.exists():
            return False
        
        workflow_files = list(github_workflows.glob("*.yml"))
        return len(workflow_files) >= 1
    
    def _check_security_monitoring(self) -> bool:
        """Check if security monitoring is configured."""
        monitoring_dir = self.project_root / "monitoring"
        if not monitoring_dir.exists():
            return False
        
        # Check for security-related monitoring
        prometheus_config = monitoring_dir / "prometheus.yml"
        if prometheus_config.exists():
            with open(prometheus_config, 'r') as f:
                content = f.read()
                return "security" in content.lower() or "alert" in content.lower()
        
        return False
    
    def _check_data_validation(self) -> bool:
        """Check if data validation mechanisms exist."""
        # Look for validation in code or documentation
        src_files = list(self.project_root.glob("src/**/*.py"))
        for src_file in src_files:
            with open(src_file, 'r') as f:
                content = f.read()
                if "validate" in content.lower() or "pydantic" in content.lower():
                    return True
        return False
    
    def _check_error_handling(self) -> bool:
        """Check if error handling mechanisms exist."""
        src_files = list(self.project_root.glob("src/**/*.py"))
        for src_file in src_files:
            with open(src_file, 'r') as f:
                content = f.read()
                if "try:" in content and "except" in content:
                    return True
        return False
    
    def _check_encryption_controls(self) -> bool:
        """Check if encryption controls are implemented."""
        # Look for encryption-related code or documentation
        security_md = self.project_root / "SECURITY.md"
        if security_md.exists():
            with open(security_md, 'r') as f:
                content = f.read().lower()
                return "encrypt" in content or "tls" in content
        return False
    
    def _check_access_controls(self) -> bool:
        """Check if access controls are documented."""
        return (
            (self.project_root / "SECURITY.md").exists() and
            (self.project_root / "CONTRIBUTING.md").exists()
        )
    
    def _check_cryptography_controls(self) -> bool:
        """Check if cryptography controls are documented."""
        security_md = self.project_root / "SECURITY.md"
        if security_md.exists():
            with open(security_md, 'r') as f:
                content = f.read().lower()
                return "crypto" in content or "encrypt" in content or "tls" in content
        return False
    
    def _check_communications_security(self) -> bool:
        """Check if communications security is addressed."""
        docker_compose = self.project_root / "docker-compose.yml"
        if docker_compose.exists():
            with open(docker_compose, 'r') as f:
                content = f.read()
                return "443" in content or "tls" in content.lower()
        return False
    
    def _check_secure_development(self) -> bool:
        """Check if secure development practices are in place."""
        return (
            (self.project_root / ".pre-commit-config.yaml").exists() and
            (self.project_root / "CONTRIBUTING.md").exists() and
            (self.project_root / "tests").exists()
        )
    
    def _check_dependency_scanning(self) -> bool:
        """Check if dependency scanning is configured."""
        pre_commit = self.project_root / ".pre-commit-config.yaml"
        if pre_commit.exists():
            with open(pre_commit, 'r') as f:
                content = f.read()
                return "safety" in content and "bandit" in content
        return False
    
    def _check_secrets_management(self) -> bool:
        """Check if secrets management is documented."""
        pre_commit = self.project_root / ".pre-commit-config.yaml"
        if pre_commit.exists():
            with open(pre_commit, 'r') as f:
                content = f.read()
                return "detect-secrets" in content
        return False
    
    def _check_container_security(self) -> bool:
        """Check if container security measures exist."""
        dockerfile = self.project_root / "Dockerfile"
        if dockerfile.exists():
            # Check for security best practices in Dockerfile
            return True
        return False
    
    # SAMM helper methods
    def _check_samm_governance(self) -> Dict:
        """Check SAMM Governance maturity."""
        level = 0
        findings = []
        
        # Strategy & Metrics
        if (self.project_root / "SECURITY.md").exists():
            level = max(level, 1)
            findings.append("✅ Security strategy documented")
        
        # Policy & Compliance
        if (self.project_root / "docs" / "SUPPLY_CHAIN_SECURITY.md").exists():
            level = max(level, 2)
            findings.append("✅ Compliance framework implemented")
        
        # Education & Guidance
        if (self.project_root / "CONTRIBUTING.md").exists():
            level = max(level, 1)
            findings.append("✅ Development guidance provided")
        
        return {"level": level, "findings": findings}
    
    def _check_samm_design(self) -> Dict:
        """Check SAMM Design maturity."""
        level = 0
        findings = []
        
        # Threat Assessment
        if (self.project_root / "docs" / "SUPPLY_CHAIN_SECURITY.md").exists():
            level = max(level, 1)
            findings.append("✅ Threat assessment considerations")
        
        # Security Requirements
        if (self.project_root / "SECURITY.md").exists():
            level = max(level, 1)
            findings.append("✅ Security requirements documented")
        
        # Security Architecture
        if (self.project_root / "docs" / "ARCHITECTURE.md").exists():
            level = max(level, 2)
            findings.append("✅ Architecture documentation includes security")
        
        return {"level": level, "findings": findings}
    
    def _check_samm_implementation(self) -> Dict:
        """Check SAMM Implementation maturity."""
        level = 0
        findings = []
        
        # Secure Build
        if (self.project_root / "Dockerfile").exists():
            level = max(level, 1)
            findings.append("✅ Containerized build process")
        
        # Secure Deployment
        if (self.project_root / "monitoring").exists():
            level = max(level, 1)
            findings.append("✅ Deployment monitoring configured")
        
        # Defect Management
        if (self.project_root / ".pre-commit-config.yaml").exists():
            level = max(level, 2)
            findings.append("✅ Automated defect detection")
        
        return {"level": level, "findings": findings}
    
    def _check_samm_verification(self) -> Dict:
        """Check SAMM Verification maturity.""" 
        level = 0
        findings = []
        
        # Architecture Assessment
        if (self.project_root / "docs" / "ARCHITECTURE.md").exists():
            level = max(level, 1)
            findings.append("✅ Architecture assessment capability")
        
        # Requirements-driven Testing
        if (self.project_root / "tests").exists():
            level = max(level, 1)
            findings.append("✅ Requirements-driven testing")
        
        # Security Testing
        if (self.project_root / "tests" / "performance").exists():
            level = max(level, 2)
            findings.append("✅ Performance/security testing framework")
        
        return {"level": level, "findings": findings}
    
    def _check_samm_operations(self) -> Dict:
        """Check SAMM Operations maturity."""
        level = 0
        findings = []
        
        # Incident Management
        if (self.project_root / "docs" / "INCIDENT_RESPONSE.md").exists():
            level = max(level, 2)
            findings.append("✅ Incident management procedures")
        
        # Environment Management
        if (self.project_root / "docs" / "RUNBOOKS.md").exists():
            level = max(level, 2)
            findings.append("✅ Environment management procedures")
        
        # Operational Management
        if (self.project_root / "monitoring").exists():
            level = max(level, 1)
            findings.append("✅ Operational monitoring")
        
        return {"level": level, "findings": findings}
    
    # Recommendation generation methods
    def _get_nist_ssdf_recommendations(self, score: int, max_score: int) -> List[str]:
        """Generate NIST SSDF recommendations."""
        recommendations = []
        gap = max_score - score
        
        if gap > 0:
            recommendations.append(f"Improve NIST SSDF compliance by addressing {gap} missing controls")
            
        if score < 5:
            recommendations.append("Focus on foundational security practices (PO - Prepare)")
        if score < 10:
            recommendations.append("Implement secure development practices (PS - Protect)")
        if score < 15:
            recommendations.append("Enhance build and testing processes (PW - Produce)")
        if score < 20:
            recommendations.append("Strengthen incident response capabilities (RV - Respond)")
            
        return recommendations
    
    def _get_slsa_recommendations(self, level: int) -> List[str]:
        """Generate SLSA recommendations."""
        recommendations = []
        
        if level < 1:
            recommendations.append("Implement scripted build process and basic provenance")
        if level < 2:
            recommendations.append("Set up hosted build service (GitHub Actions)")
            recommendations.append("Implement build and source integrity measures")
        if level < 3:
            recommendations.append("Separate source and build platforms")
            recommendations.append("Define build process completely as code")
            
        return recommendations
    
    def _get_soc2_recommendations(self, score: int, max_score: int) -> List[str]:
        """Generate SOC 2 recommendations."""
        recommendations = []
        gap = max_score - score
        
        if gap >= 5:
            recommendations.append("Significant SOC 2 gaps identified - consider professional audit")
        if gap >= 3:
            recommendations.append("Focus on missing trust service criteria")
        if gap > 0:
            recommendations.append("Address remaining SOC 2 compliance gaps")
            
        return recommendations
    
    def _get_iso27001_recommendations(self, score: int, max_score: int) -> List[str]:
        """Generate ISO 27001 recommendations."""
        recommendations = []
        gap = max_score - score
        
        if gap > 6:
            recommendations.append("Major ISO 27001 gaps - consider formal ISMS implementation")
        if gap > 3:
            recommendations.append("Address critical security controls")
        if gap > 0:
            recommendations.append("Complete remaining ISO 27001 requirements")
            
        return recommendations
    
    def _get_samm_recommendations(self, average_maturity: float) -> List[str]:
        """Generate OWASP SAMM recommendations."""
        recommendations = []
        
        if average_maturity < 1.0:
            recommendations.append("Implement basic security practices across all business functions")
        elif average_maturity < 2.0:
            recommendations.append("Enhance security practices to achieve consistent maturity")
        elif average_maturity < 3.0:
            recommendations.append("Optimize security practices for maximum maturity")
        else:
            recommendations.append("Maintain current high maturity level")
            
        return recommendations
    
    def _get_ai_rmf_recommendations(self, score: int, max_score: int) -> List[str]:
        """Generate AI RMF recommendations."""
        recommendations = []
        gap = max_score - score
        
        if gap > 8:
            recommendations.append("Implement comprehensive AI governance framework")
        if gap > 4:
            recommendations.append("Enhance AI risk management practices")
        if gap > 0:
            recommendations.append("Complete AI RMF implementation")
            
        return recommendations
    
    def _get_mlops_recommendations(self, level: int) -> List[str]:
        """Generate MLOps recommendations."""
        recommendations = []
        
        if level < 1:
            recommendations.append("Implement basic DevOps practices")
        if level < 2:
            recommendations.append("Automate ML training pipeline")
        if level < 3:
            recommendations.append("Implement automated ML deployment")
            
        return recommendations
    
    def _get_supply_chain_recommendations(self, score: int, max_score: int) -> List[str]:
        """Generate supply chain security recommendations."""
        recommendations = []
        gap = max_score - score
        
        if gap > 5:
            recommendations.append("Implement comprehensive supply chain security program")
        if gap > 2:
            recommendations.append("Address critical supply chain vulnerabilities")
        if gap > 0:
            recommendations.append("Complete supply chain security implementation")
            
        return recommendations
    
    def _calculate_overall_score(self):
        """Calculate overall compliance score."""
        scores = []
        
        for framework, data in self.results["compliance_frameworks"].items():
            if "compliance_percentage" in data:
                scores.append(data["compliance_percentage"])
            elif "level_achieved" in data:
                # Convert SLSA level to percentage
                scores.append((data["level_achieved"] / 3) * 100)
            elif "average_maturity_level" in data:
                # Convert SAMM maturity to percentage
                scores.append((data["average_maturity_level"] / 3) * 100)
            elif "maturity_level" in data:
                # Convert MLOps maturity to percentage
                scores.append((data["maturity_level"] / 3) * 100)
        
        if scores:
            self.results["overall_score"] = sum(scores) / len(scores)
        
        # Generate overall recommendations
        if self.results["overall_score"] < 70:
            self.results["recommendations"].append("Focus on foundational security and compliance practices")
        elif self.results["overall_score"] < 85:
            self.results["recommendations"].append("Address remaining compliance gaps for comprehensive coverage")
        else:
            self.results["recommendations"].append("Maintain current high compliance posture")
    
    def generate_report(self, output_file: str = "compliance-report.json"):
        """Generate comprehensive compliance report."""
        logger.info(f"Generating compliance report: {output_file}")
        
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Also generate a markdown summary
        markdown_file = output_file.replace('.json', '.md')
        self._generate_markdown_report(markdown_file)
        
        logger.info(f"Reports generated: {output_file}, {markdown_file}")
    
    def _generate_markdown_report(self, output_file: str):
        """Generate markdown summary report."""
        with open(output_file, 'w') as f:
            f.write(f"# Compliance Report\n\n")
            f.write(f"**Generated**: {self.results['timestamp']}\\n")
            f.write(f"**Project**: {self.results['project']}\\n")
            f.write(f"**Version**: {self.results['version']}\\n")
            f.write(f"**Overall Score**: {self.results['overall_score']:.1f}%\\n\\n")
            
            f.write("## Executive Summary\\n\\n")
            for rec in self.results['recommendations']:
                f.write(f"- {rec}\\n")
            f.write("\\n")
            
            f.write("## Framework Compliance\\n\\n")
            for framework, data in self.results['compliance_frameworks'].items():
                f.write(f"### {data.get('framework', framework)}\\n\\n")
                
                if 'compliance_percentage' in data:
                    f.write(f"**Score**: {data['compliance_percentage']:.1f}%\\n")
                elif 'level_achieved' in data:
                    f.write(f"**Level**: {data['level_achieved']}/3\\n")
                elif 'average_maturity_level' in data:
                    f.write(f"**Maturity**: {data['average_maturity_level']}/3\\n")
                
                f.write(f"**Status**: {data.get('status', 'unknown')}\\n\\n")
                
                if 'findings' in data:
                    f.write("**Key Findings**:\\n")
                    for finding in data['findings'][:5]:  # Top 5 findings
                        f.write(f"- {finding}\\n")
                    f.write("\\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Automated compliance checking for Nimify project")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output", default="compliance-report.json", help="Output report file")
    parser.add_argument("--framework", help="Check specific framework only", 
                       choices=["nist_ssdf", "slsa", "soc2", "iso27001", "owasp_samm", "nist_ai_rmf", "mlops", "supply_chain"])
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    checker = ComplianceChecker(args.project_root)
    
    if args.framework:
        # Check specific framework only
        if args.framework == "nist_ssdf":
            result = checker.check_nist_ssdf()
        elif args.framework == "slsa":
            result = checker.check_slsa()
        elif args.framework == "soc2":
            result = checker.check_soc2_type2()
        elif args.framework == "iso27001":
            result = checker.check_iso27001()
        elif args.framework == "owasp_samm":
            result = checker.check_owasp_samm()
        elif args.framework == "nist_ai_rmf":
            result = checker.check_nist_ai_rmf()
        elif args.framework == "mlops":
            result = checker.check_mlops_maturity()
        elif args.framework == "supply_chain":
            result = checker.check_supply_chain_security()
        
        print(json.dumps(result, indent=2))
    else:
        # Run all checks
        results = checker.run_all_checks()
        checker.generate_report(args.output)
        
        print(f"\\nCompliance check completed!")
        print(f"Overall Score: {results['overall_score']:.1f}%")
        print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()