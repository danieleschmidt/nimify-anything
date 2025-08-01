#\!/usr/bin/env python3
"""
Compliance checking script for Nimify Anything.
Verifies license compliance, SLSA requirements, and generates compliance reports.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any
import subprocess


class ComplianceChecker:
    """Main compliance checking class."""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.results = {
            'licenses': {'passed': False, 'issues': []},
            'slsa': {'passed': False, 'issues': []},
            'summary': {'total_checks': 0, 'passed': 0, 'failed': 0}
        }
    
    def check_licenses(self) -> bool:
        """Check license compliance."""
        print("🔍 Checking license compliance...")
        
        # License header patterns
        license_patterns = [
            r'Copyright \(c\) \d{4}',
            r'MIT License',
            r'Licensed under the MIT License',
            r'SPDX-License-Identifier: MIT'
        ]
        
        # Files that should have license headers
        source_files = []
        for pattern in ['**/*.py', '**/*.js', '**/*.ts', '**/*.go', '**/*.java']:
            source_files.extend(self.project_root.glob(pattern))
        
        # Exclude certain directories
        exclude_dirs = {'.git', '__pycache__', 'node_modules', '.venv', 'venv', 'build', 'dist'}
        source_files = [f for f in source_files if not any(excl in f.parts for excl in exclude_dirs)]
        
        missing_license = []
        for file_path in source_files:
            try:
                content = file_path.read_text(encoding='utf-8')[:1000]  # Check first 1000 chars
                
                # Skip empty files or test files
                if len(content.strip()) < 10 or 'test_' in file_path.name:
                    continue
                
                has_license = any(re.search(pattern, content, re.IGNORECASE) for pattern in license_patterns)
                
                if not has_license:
                    missing_license.append(str(file_path.relative_to(self.project_root)))
            
            except (UnicodeDecodeError, IOError):
                continue  # Skip binary or unreadable files
        
        if missing_license:
            self.results['licenses']['issues'] = missing_license
            self.results['licenses']['passed'] = False
            print(f"❌ {len(missing_license)} files missing license headers")
            for file in missing_license[:10]:  # Show first 10
                print(f"   - {file}")
            if len(missing_license) > 10:
                print(f"   ... and {len(missing_license) - 10} more")
        else:
            self.results['licenses']['passed'] = True
            print("✅ All source files have proper license headers")
        
        return self.results['licenses']['passed']
    
    def check_slsa(self) -> bool:
        """Check SLSA (Supply-chain Levels for Software Artifacts) compliance."""
        print("🔍 Checking SLSA compliance...")
        
        issues = []
        
        # Check for build automation
        github_workflows = self.project_root / '.github' / 'workflows'
        if not github_workflows.exists():
            issues.append("No GitHub Actions workflows found")
        else:
            ci_files = list(github_workflows.glob('*.yml')) + list(github_workflows.glob('*.yaml'))
            if not ci_files:
                issues.append("No CI/CD workflows configured")
        
        # Check for reproducible builds
        dockerfile_files = list(self.project_root.glob('Dockerfile*'))
        if dockerfile_files:
            for dockerfile in dockerfile_files:
                content = dockerfile.read_text()
                if 'pip install' in content and '--no-cache-dir' not in content:
                    issues.append(f"Dockerfile {dockerfile.name} should use --no-cache-dir for reproducible builds")
        
        # Check for SBOM generation
        if not any('sbom' in f.name.lower() for f in github_workflows.glob('*.yml') if github_workflows.exists()):
            issues.append("No SBOM (Software Bill of Materials) generation configured")
        
        # Check for security scanning
        security_files = [f for f in github_workflows.glob('*.yml') if 'security' in f.name.lower()] if github_workflows.exists() else []
        if not security_files:
            issues.append("No automated security scanning configured")
        
        if issues:
            self.results['slsa']['issues'] = issues
            self.results['slsa']['passed'] = False
            print(f"❌ SLSA compliance issues found:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            self.results['slsa']['passed'] = True
            print("✅ SLSA compliance requirements met")
        
        return self.results['slsa']['passed']
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate compliance report."""
        print("📋 Generating compliance report...")
        
        # Calculate summary
        total_checks = 2  # license + slsa
        passed_checks = sum([
            1 if self.results['licenses']['passed'] else 0,
            1 if self.results['slsa']['passed'] else 0
        ])
        
        self.results['summary'] = {
            'total_checks': total_checks,
            'passed': passed_checks,
            'failed': total_checks - passed_checks,
            'compliance_score': (passed_checks / total_checks) * 100
        }
        
        # Add metadata
        self.results['metadata'] = {
            'timestamp': subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ']).decode().strip(),
            'repository': os.environ.get('GITHUB_REPOSITORY', 'unknown'),
            'commit': os.environ.get('GITHUB_SHA', subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip() if self._git_available() else 'unknown'),
            'branch': os.environ.get('GITHUB_REF_NAME', subprocess.check_output(['git', 'branch', '--show-current']).decode().strip() if self._git_available() else 'unknown')
        }
        
        # Write report
        report_file = self.project_root / 'compliance-report.json'
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"📄 Compliance report written to {report_file}")
        print(f"📊 Compliance Score: {self.results['summary']['compliance_score']:.1f}%")
        
        return self.results
    
    def _git_available(self) -> bool:
        """Check if git is available."""
        try:
            subprocess.check_output(['git', '--version'], stderr=subprocess.DEVNULL)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Compliance checker for Nimify Anything')
    parser.add_argument('--check-licenses', action='store_true', help='Check license compliance')
    parser.add_argument('--check-slsa', action='store_true', help='Check SLSA compliance')
    parser.add_argument('--generate-report', action='store_true', help='Generate compliance report')
    parser.add_argument('--all', action='store_true', help='Run all checks and generate report')
    parser.add_argument('--project-root', type=Path, help='Project root directory')
    
    args = parser.parse_args()
    
    # If no specific checks requested, run all
    if not any([args.check_licenses, args.check_slsa, args.generate_report]):
        args.all = True
    
    checker = ComplianceChecker(args.project_root)
    
    exit_code = 0
    
    try:
        if args.all or args.check_licenses:
            if not checker.check_licenses():
                exit_code = 1
        
        if args.all or args.check_slsa:
            if not checker.check_slsa():
                exit_code = 1
        
        if args.all or args.generate_report:
            checker.generate_report()
        
        # Print summary
        if args.all:
            summary = checker.results['summary']
            print("\n" + "="*50)
            print("COMPLIANCE SUMMARY")
            print("="*50)
            print(f"Total Checks: {summary['total_checks']}")
            print(f"Passed: {summary['passed']}")
            print(f"Failed: {summary['failed']}")
            print(f"Score: {summary['compliance_score']:.1f}%")
            
            if summary['failed'] > 0:
                print("\n⚠️  Compliance issues found. Please address them before production deployment.")
            else:
                print("\n✅ All compliance checks passed\!")
    
    except Exception as e:
        print(f"❌ Error during compliance check: {e}")
        exit_code = 1
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
EOF < /dev/null
