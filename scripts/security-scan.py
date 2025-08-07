#!/usr/bin/env python3
"""Security scanning script for Nimify codebase."""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any

def run_bandit_scan() -> Dict[str, Any]:
    """Run Bandit security scanner."""
    print("🔍 Running Bandit security scan...")
    
    try:
        result = subprocess.run([
            "bandit", "-r", "src/", 
            "-f", "json", 
            "-o", "/tmp/bandit-report.json"
        ], capture_output=True, text=True, check=False)
        
        if Path("/tmp/bandit-report.json").exists():
            with open("/tmp/bandit-report.json", "r") as f:
                report = json.load(f)
            
            return {
                "tool": "bandit",
                "status": "completed",
                "issues": len(report.get("results", [])),
                "high_severity": len([r for r in report.get("results", []) if r.get("issue_severity") == "HIGH"]),
                "medium_severity": len([r for r in report.get("results", []) if r.get("issue_severity") == "MEDIUM"]),
                "report_path": "/tmp/bandit-report.json"
            }
        else:
            return {"tool": "bandit", "status": "failed", "error": result.stderr}
            
    except FileNotFoundError:
        return {"tool": "bandit", "status": "not_installed", "error": "Bandit not found"}


def run_safety_scan() -> Dict[str, Any]:
    """Run Safety vulnerability scanner for dependencies."""
    print("🔍 Running Safety dependency scan...")
    
    try:
        result = subprocess.run([
            "safety", "check", "--json"
        ], capture_output=True, text=True, check=False)
        
        if result.returncode == 0:
            return {
                "tool": "safety",
                "status": "completed", 
                "vulnerabilities": 0,
                "message": "No known vulnerabilities found"
            }
        else:
            # Parse JSON output if available
            try:
                report = json.loads(result.stdout)
                return {
                    "tool": "safety",
                    "status": "completed",
                    "vulnerabilities": len(report),
                    "details": report
                }
            except json.JSONDecodeError:
                return {
                    "tool": "safety", 
                    "status": "completed",
                    "vulnerabilities": "unknown",
                    "output": result.stdout
                }
                
    except FileNotFoundError:
        return {"tool": "safety", "status": "not_installed", "error": "Safety not found"}


def run_secrets_scan() -> Dict[str, Any]:
    """Run detect-secrets scanner."""
    print("🔍 Running secrets detection scan...")
    
    try:
        # Generate baseline
        subprocess.run([
            "detect-secrets", "scan", "--all-files", 
            "--baseline", "/tmp/secrets-baseline.json"
        ], capture_output=True, check=False)
        
        # Audit against baseline
        result = subprocess.run([
            "detect-secrets", "audit", "/tmp/secrets-baseline.json"
        ], capture_output=True, text=True, check=False)
        
        if Path("/tmp/secrets-baseline.json").exists():
            with open("/tmp/secrets-baseline.json", "r") as f:
                baseline = json.load(f)
            
            secrets_count = sum(len(files) for files in baseline.get("results", {}).values())
            
            return {
                "tool": "detect-secrets",
                "status": "completed",
                "potential_secrets": secrets_count,
                "baseline_path": "/tmp/secrets-baseline.json"
            }
        else:
            return {"tool": "detect-secrets", "status": "failed", "error": "No baseline generated"}
            
    except FileNotFoundError:
        return {"tool": "detect-secrets", "status": "not_installed", "error": "detect-secrets not found"}


def run_dockerfile_scan() -> Dict[str, Any]:
    """Scan Dockerfiles for security issues."""
    print("🔍 Scanning Dockerfiles...")
    
    dockerfiles = list(Path(".").glob("**/Dockerfile*"))
    
    if not dockerfiles:
        return {"tool": "dockerfile_scan", "status": "no_dockerfiles", "message": "No Dockerfiles found"}
    
    issues = []
    
    for dockerfile in dockerfiles:
        try:
            with open(dockerfile, "r") as f:
                content = f.read()
            
            # Basic security checks
            file_issues = []
            
            if "USER root" in content or "USER 0" in content:
                file_issues.append("Running as root user")
            
            if "chmod 777" in content:
                file_issues.append("Overly permissive file permissions")
            
            if "ADD http" in content:
                file_issues.append("Using ADD with URL (use COPY instead)")
            
            if "--no-cache-dir" not in content and "pip install" in content:
                file_issues.append("pip install without --no-cache-dir")
            
            if file_issues:
                issues.append({"file": str(dockerfile), "issues": file_issues})
        
        except Exception as e:
            issues.append({"file": str(dockerfile), "error": str(e)})
    
    return {
        "tool": "dockerfile_scan",
        "status": "completed",
        "files_scanned": len(dockerfiles),
        "issues": issues
    }


def check_file_permissions() -> Dict[str, Any]:
    """Check for overly permissive file permissions."""
    print("🔍 Checking file permissions...")
    
    issues = []
    
    for root, dirs, files in os.walk("src/"):
        for file in files:
            file_path = Path(root) / file
            try:
                stat = file_path.stat()
                mode = oct(stat.st_mode)[-3:]
                
                # Check for world-writable files
                if mode.endswith('2') or mode.endswith('3') or mode.endswith('6') or mode.endswith('7'):
                    issues.append({
                        "file": str(file_path),
                        "permissions": mode,
                        "issue": "World-writable file"
                    })
                
            except Exception as e:
                issues.append({"file": str(file_path), "error": str(e)})
    
    return {
        "tool": "file_permissions",
        "status": "completed",
        "issues": issues
    }


def generate_security_report(scan_results: List[Dict[str, Any]]) -> None:
    """Generate comprehensive security report."""
    print("\n" + "="*60)
    print("🛡️  SECURITY SCAN RESULTS")
    print("="*60)
    
    total_issues = 0
    critical_issues = 0
    
    for result in scan_results:
        tool = result.get("tool", "unknown")
        status = result.get("status", "unknown")
        
        print(f"\n📊 {tool.upper()}")
        print("-" * 40)
        
        if status == "completed":
            if tool == "bandit":
                issues = result.get("issues", 0)
                high = result.get("high_severity", 0)
                medium = result.get("medium_severity", 0)
                
                print(f"  Total Issues: {issues}")
                print(f"  High Severity: {high}")
                print(f"  Medium Severity: {medium}")
                
                total_issues += issues
                critical_issues += high
                
                if high > 0:
                    print(f"  ⚠️  Report: {result.get('report_path', 'N/A')}")
            
            elif tool == "safety":
                vulns = result.get("vulnerabilities", 0)
                print(f"  Vulnerabilities: {vulns}")
                
                total_issues += vulns if isinstance(vulns, int) else 0
                
                if vulns > 0:
                    critical_issues += vulns if isinstance(vulns, int) else 0
            
            elif tool == "detect-secrets":
                secrets = result.get("potential_secrets", 0)
                print(f"  Potential Secrets: {secrets}")
                
                total_issues += secrets
                if secrets > 0:
                    critical_issues += secrets
                    print(f"  ⚠️  Baseline: {result.get('baseline_path', 'N/A')}")
            
            elif tool == "dockerfile_scan":
                files = result.get("files_scanned", 0)
                issues = result.get("issues", [])
                
                print(f"  Files Scanned: {files}")
                print(f"  Issues Found: {len(issues)}")
                
                total_issues += len(issues)
                
                for issue in issues:
                    if "issues" in issue:
                        print(f"    {issue['file']}: {', '.join(issue['issues'])}")
            
            elif tool == "file_permissions":
                issues = result.get("issues", [])
                print(f"  Permission Issues: {len(issues)}")
                
                total_issues += len(issues)
                critical_issues += len(issues)
                
                for issue in issues:
                    if "issue" in issue:
                        print(f"    {issue['file']}: {issue['issue']} ({issue['permissions']})")
        
        elif status == "not_installed":
            print(f"  ❌ Tool not installed: {result.get('error', 'Unknown error')}")
        
        elif status == "failed":
            print(f"  ❌ Scan failed: {result.get('error', 'Unknown error')}")
        
        else:
            print(f"  ℹ️  {result.get('message', 'No additional info')}")
    
    # Summary
    print("\n" + "="*60)
    print("📋 SECURITY SUMMARY")
    print("="*60)
    print(f"Total Issues: {total_issues}")
    print(f"Critical Issues: {critical_issues}")
    
    if critical_issues > 0:
        print("🚨 CRITICAL: Immediate attention required!")
        sys.exit(1)
    elif total_issues > 0:
        print("⚠️  WARNING: Security issues found")
        sys.exit(1)
    else:
        print("✅ PASSED: No critical security issues detected")
        sys.exit(0)


def main():
    """Main security scanning function."""
    print("🛡️  Starting comprehensive security scan...")
    
    scan_results = [
        run_bandit_scan(),
        run_safety_scan(), 
        run_secrets_scan(),
        run_dockerfile_scan(),
        check_file_permissions()
    ]
    
    generate_security_report(scan_results)


if __name__ == "__main__":
    main()