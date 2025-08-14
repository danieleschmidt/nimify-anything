"""Comprehensive Quality Assurance Report Generator."""

import os
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import ast
import re


@dataclass
class QualityMetric:
    """Individual quality metric."""
    name: str
    value: float
    threshold: float
    passed: bool
    description: str
    severity: str = "medium"  # low, medium, high, critical


@dataclass
class QualityGateResult:
    """Result from a quality gate check."""
    gate_name: str
    passed: bool
    metrics: List[QualityMetric]
    execution_time: float
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]


class CodeQualityAnalyzer:
    """Analyzes code quality without external dependencies."""
    
    def __init__(self, source_dir: str = "src"):
        self.source_dir = Path(source_dir)
        self.results = []
    
    def analyze_code_structure(self) -> QualityGateResult:
        """Analyze code structure and organization."""
        start_time = time.time()
        metrics = []
        errors = []
        warnings = []
        recommendations = []
        
        try:
            # Count Python files
            python_files = list(self.source_dir.rglob("*.py"))
            total_files = len(python_files)
            
            # Analyze file sizes
            file_sizes = []
            for file_path in python_files:
                try:
                    size = file_path.stat().st_size
                    file_sizes.append(size)
                    
                    # Check for very large files
                    if size > 10000:  # 10KB threshold
                        warnings.append(f"Large file detected: {file_path} ({size} bytes)")
                        
                except Exception as e:
                    errors.append(f"Failed to analyze {file_path}: {e}")
            
            # Calculate metrics
            avg_file_size = sum(file_sizes) / len(file_sizes) if file_sizes else 0
            max_file_size = max(file_sizes) if file_sizes else 0
            
            metrics.extend([
                QualityMetric(
                    name="Total Python Files",
                    value=total_files,
                    threshold=5,
                    passed=total_files >= 5,
                    description="Number of Python source files"
                ),
                QualityMetric(
                    name="Average File Size",
                    value=avg_file_size,
                    threshold=5000,
                    passed=avg_file_size <= 5000,
                    description="Average file size in bytes"
                ),
                QualityMetric(
                    name="Max File Size",
                    value=max_file_size,
                    threshold=20000,
                    passed=max_file_size <= 20000,
                    description="Maximum file size in bytes",
                    severity="medium"
                )
            ])
            
            # Analyze import structure
            import_violations = self._analyze_imports(python_files)
            
            metrics.append(QualityMetric(
                name="Import Violations",
                value=len(import_violations),
                threshold=5,
                passed=len(import_violations) <= 5,
                description="Number of problematic imports",
                severity="high"
            ))
            
            warnings.extend(import_violations)
            
            # Check for common code smells
            code_smells = self._detect_code_smells(python_files)
            
            metrics.append(QualityMetric(
                name="Code Smells",
                value=len(code_smells),
                threshold=10,
                passed=len(code_smells) <= 10,
                description="Number of detected code smells",
                severity="medium"
            ))
            
            warnings.extend(code_smells)
            
            if avg_file_size > 3000:
                recommendations.append("Consider breaking large files into smaller modules")
            
            if total_files > 50:
                recommendations.append("Consider organizing code into packages")
                
        except Exception as e:
            errors.append(f"Code structure analysis failed: {e}")
        
        execution_time = time.time() - start_time
        passed = all(metric.passed for metric in metrics) and len(errors) == 0
        
        return QualityGateResult(
            gate_name="Code Structure",
            passed=passed,
            metrics=metrics,
            execution_time=execution_time,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _analyze_imports(self, python_files: List[Path]) -> List[str]:
        """Analyze import statements for issues."""
        violations = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST
                try:
                    tree = ast.parse(content)
                except SyntaxError:
                    violations.append(f"Syntax error in {file_path}")
                    continue
                
                # Check imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.startswith('.'):
                                violations.append(f"Relative import in {file_path}: {alias.name}")
                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.level > 2:  # More than two levels of relative imports
                            violations.append(f"Deep relative import in {file_path}")
                        
                        # Check for wildcard imports
                        for alias in node.names:
                            if alias.name == '*':
                                violations.append(f"Wildcard import in {file_path}")
                                
            except Exception as e:
                violations.append(f"Failed to analyze imports in {file_path}: {e}")
        
        return violations
    
    def _detect_code_smells(self, python_files: List[Path]) -> List[str]:
        """Detect common code smells."""
        smells = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                # Check for long lines
                for i, line in enumerate(lines, 1):
                    if len(line) > 120:
                        smells.append(f"Long line in {file_path}:{i} ({len(line)} chars)")
                
                # Check for TODO/FIXME comments
                todo_pattern = re.compile(r'#\s*(TODO|FIXME|XXX|HACK)', re.IGNORECASE)
                for i, line in enumerate(lines, 1):
                    if todo_pattern.search(line):
                        smells.append(f"TODO/FIXME comment in {file_path}:{i}")
                
                # Check for deeply nested code
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            depth = self._calculate_nesting_depth(node)
                            if depth > 4:
                                smells.append(f"Deep nesting in function {node.name} in {file_path} (depth: {depth})")
                except SyntaxError:
                    pass
                    
            except Exception as e:
                smells.append(f"Failed to analyze {file_path}: {e}")
        
        return smells
    
    def _calculate_nesting_depth(self, node) -> int:
        """Calculate maximum nesting depth in a function."""
        max_depth = 0
        
        def traverse(node, depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                    traverse(child, depth + 1)
                else:
                    traverse(child, depth)
        
        traverse(node)
        return max_depth
    
    def analyze_documentation(self) -> QualityGateResult:
        """Analyze documentation quality."""
        start_time = time.time()
        metrics = []
        errors = []
        warnings = []
        recommendations = []
        
        try:
            python_files = list(self.source_dir.rglob("*.py"))
            
            documented_functions = 0
            total_functions = 0
            documented_classes = 0
            total_classes = 0
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            total_functions += 1
                            if ast.get_docstring(node):
                                documented_functions += 1
                        
                        elif isinstance(node, ast.ClassDef):
                            total_classes += 1
                            if ast.get_docstring(node):
                                documented_classes += 1
                                
                except Exception as e:
                    errors.append(f"Failed to analyze documentation in {file_path}: {e}")
            
            # Calculate documentation coverage
            func_doc_coverage = (documented_functions / total_functions * 100) if total_functions > 0 else 0
            class_doc_coverage = (documented_classes / total_classes * 100) if total_classes > 0 else 0
            
            metrics.extend([
                QualityMetric(
                    name="Function Documentation Coverage",
                    value=func_doc_coverage,
                    threshold=70.0,
                    passed=func_doc_coverage >= 70.0,
                    description="Percentage of functions with docstrings",
                    severity="medium"
                ),
                QualityMetric(
                    name="Class Documentation Coverage",
                    value=class_doc_coverage,
                    threshold=80.0,
                    passed=class_doc_coverage >= 80.0,
                    description="Percentage of classes with docstrings",
                    severity="medium"
                )
            ])
            
            # Check for README files
            readme_files = [
                self.source_dir.parent / "README.md",
                self.source_dir.parent / "README.rst",
                self.source_dir.parent / "README.txt"
            ]
            
            has_readme = any(readme.exists() for readme in readme_files)
            
            metrics.append(QualityMetric(
                name="README Exists",
                value=1.0 if has_readme else 0.0,
                threshold=1.0,
                passed=has_readme,
                description="Project has a README file",
                severity="high"
            ))
            
            if func_doc_coverage < 50:
                recommendations.append("Add docstrings to functions for better maintainability")
            
            if class_doc_coverage < 60:
                recommendations.append("Add docstrings to classes to improve code documentation")
                
        except Exception as e:
            errors.append(f"Documentation analysis failed: {e}")
        
        execution_time = time.time() - start_time
        passed = all(metric.passed for metric in metrics) and len(errors) == 0
        
        return QualityGateResult(
            gate_name="Documentation",
            passed=passed,
            metrics=metrics,
            execution_time=execution_time,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def analyze_security(self) -> QualityGateResult:
        """Analyze security-related issues."""
        start_time = time.time()
        metrics = []
        errors = []
        warnings = []
        recommendations = []
        
        try:
            python_files = list(self.source_dir.rglob("*.py"))
            security_issues = []
            
            # Security patterns to check
            security_patterns = [
                (r'exec\s*\(', "Use of exec() function"),
                (r'eval\s*\(', "Use of eval() function"),
                (r'os\.system\s*\(', "Use of os.system()"),
                (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', "Subprocess with shell=True"),
                (r'input\s*\([^)]*\).*exec', "Input directly to exec"),
                (r'pickle\.loads?\s*\(', "Use of pickle.load/loads"),
                (r'yaml\.load\s*\((?!.*Loader)', "Unsafe YAML loading"),
                (r'hashlib\.md5\s*\(', "Use of MD5 hash (weak)"),
                (r'hashlib\.sha1\s*\(', "Use of SHA1 hash (weak)"),
                (r'random\.random\s*\(', "Use of random for security (use secrets module)"),
                (r'["\']password["\']', "Hardcoded password reference"),
                (r'["\']secret["\']', "Hardcoded secret reference"),
                (r'["\']key["\'].*=', "Potential hardcoded key"),
            ]
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    for pattern, description in security_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            security_issues.append(f"{description} in {file_path}:{line_num}")
                            
                except Exception as e:
                    errors.append(f"Failed to analyze security in {file_path}: {e}")
            
            metrics.extend([
                QualityMetric(
                    name="Security Issues",
                    value=len(security_issues),
                    threshold=0,
                    passed=len(security_issues) == 0,
                    description="Number of potential security issues",
                    severity="critical"
                ),
                QualityMetric(
                    name="High-Risk Patterns",
                    value=len([issue for issue in security_issues if any(pattern in issue for pattern in ["exec(", "eval(", "shell=True"])]),
                    threshold=0,
                    passed=not any(pattern in str(security_issues) for pattern in ["exec(", "eval(", "shell=True"]),
                    description="Number of high-risk security patterns",
                    severity="critical"
                )
            ])
            
            warnings.extend(security_issues)
            
            if security_issues:
                recommendations.append("Review and remediate security issues before production deployment")
                recommendations.append("Consider using static security analysis tools like bandit")
                
        except Exception as e:
            errors.append(f"Security analysis failed: {e}")
        
        execution_time = time.time() - start_time
        passed = all(metric.passed for metric in metrics) and len(errors) == 0
        
        return QualityGateResult(
            gate_name="Security",
            passed=passed,
            metrics=metrics,
            execution_time=execution_time,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def analyze_dependencies(self) -> QualityGateResult:
        """Analyze project dependencies."""
        start_time = time.time()
        metrics = []
        errors = []
        warnings = []
        recommendations = []
        
        try:
            # Check for pyproject.toml
            pyproject_path = self.source_dir.parent / "pyproject.toml"
            requirements_path = self.source_dir.parent / "requirements.txt"
            
            has_dependency_file = pyproject_path.exists() or requirements_path.exists()
            
            metrics.append(QualityMetric(
                name="Dependency File Exists",
                value=1.0 if has_dependency_file else 0.0,
                threshold=1.0,
                passed=has_dependency_file,
                description="Project has dependency specification",
                severity="high"
            ))
            
            # Analyze imports vs declared dependencies
            python_files = list(self.source_dir.rglob("*.py"))
            imported_modules = set()
            
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                imported_modules.add(alias.name.split('.')[0])
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                imported_modules.add(node.module.split('.')[0])
                                
                except Exception as e:
                    errors.append(f"Failed to analyze imports in {file_path}: {e}")
            
            # Filter out standard library modules
            stdlib_modules = {
                'os', 'sys', 'time', 'json', 'logging', 'threading', 'asyncio',
                'collections', 'dataclasses', 'enum', 'pathlib', 'typing',
                'unittest', 'subprocess', 'hashlib', 'functools', 'itertools',
                'datetime', 'math', 'random', 'string', 'uuid', 'warnings',
                'weakref', 'gc', 'traceback', 'copy', 'pickle', 'tempfile',
                'shutil', 'glob', 'fnmatch', 'contextlib', 'abc', 'io',
                'csv', 'sqlite3', 'urllib', 'http', 'xml', 're'
            }
            
            third_party_imports = imported_modules - stdlib_modules
            
            metrics.append(QualityMetric(
                name="Third-party Dependencies",
                value=len(third_party_imports),
                threshold=20,
                passed=len(third_party_imports) <= 20,
                description="Number of third-party dependencies",
                severity="medium"
            ))
            
            if len(third_party_imports) > 15:
                recommendations.append("Consider reducing the number of dependencies")
            
            if not has_dependency_file:
                recommendations.append("Add a dependency specification file (pyproject.toml or requirements.txt)")
                
        except Exception as e:
            errors.append(f"Dependency analysis failed: {e}")
        
        execution_time = time.time() - start_time
        passed = all(metric.passed for metric in metrics) and len(errors) == 0
        
        return QualityGateResult(
            gate_name="Dependencies",
            passed=passed,
            metrics=metrics,
            execution_time=execution_time,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def run_all_checks(self) -> List[QualityGateResult]:
        """Run all quality gate checks."""
        checks = [
            self.analyze_code_structure,
            self.analyze_documentation,
            self.analyze_security,
            self.analyze_dependencies
        ]
        
        results = []
        for check in checks:
            try:
                result = check()
                results.append(result)
            except Exception as e:
                # Create a failed result for the check
                results.append(QualityGateResult(
                    gate_name=check.__name__.replace('analyze_', '').title(),
                    passed=False,
                    metrics=[],
                    execution_time=0.0,
                    errors=[f"Check failed: {e}"],
                    warnings=[],
                    recommendations=[]
                ))
        
        return results


class QualityReportGenerator:
    """Generates comprehensive quality reports."""
    
    def __init__(self):
        self.analyzer = CodeQualityAnalyzer()
    
    def generate_report(self, output_file: str = "quality_report.json") -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        print("🔍 Running Quality Assurance Checks...")
        
        start_time = time.time()
        
        # Run all quality checks
        results = self.analyzer.run_all_checks()
        
        # Calculate overall metrics
        total_metrics = sum(len(result.metrics) for result in results)
        passed_metrics = sum(len([m for m in result.metrics if m.passed]) for result in results)
        total_errors = sum(len(result.errors) for result in results)
        total_warnings = sum(len(result.warnings) for result in results)
        
        overall_pass_rate = (passed_metrics / total_metrics * 100) if total_metrics > 0 else 0
        overall_passed = all(result.passed for result in results)
        
        # Generate summary
        summary = {
            "overall_status": "PASSED" if overall_passed else "FAILED",
            "pass_rate": overall_pass_rate,
            "total_checks": len(results),
            "passed_checks": len([r for r in results if r.passed]),
            "total_metrics": total_metrics,
            "passed_metrics": passed_metrics,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "execution_time": time.time() - start_time,
            "timestamp": time.time()
        }
        
        # Compile all recommendations
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)
        
        # Create full report
        report = {
            "summary": summary,
            "quality_gates": [asdict(result) for result in results],
            "recommendations": list(set(all_recommendations)),  # Remove duplicates
            "metadata": {
                "source_directory": str(self.analyzer.source_dir),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "generator_version": "1.0.0"
            }
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a formatted summary of the quality report."""
        summary = report["summary"]
        
        print("\n" + "="*80)
        print("🛡️  QUALITY ASSURANCE REPORT")
        print("="*80)
        
        # Overall status
        status_emoji = "✅" if summary["overall_status"] == "PASSED" else "❌"
        print(f"Overall Status: {status_emoji} {summary['overall_status']}")
        print(f"Pass Rate: {summary['pass_rate']:.1f}%")
        print(f"Execution Time: {summary['execution_time']:.2f}s")
        print()
        
        # Quality gates summary
        print("Quality Gates:")
        print("-" * 50)
        
        for gate in report["quality_gates"]:
            gate_emoji = "✅" if gate["passed"] else "❌"
            print(f"{gate_emoji} {gate['gate_name']:<20} "
                  f"({len([m for m in gate['metrics'] if m['passed']])}/{len(gate['metrics'])} metrics passed)")
            
            if gate["errors"]:
                print(f"   ❌ {len(gate['errors'])} errors")
            if gate["warnings"]:
                print(f"   ⚠️  {len(gate['warnings'])} warnings")
        
        print()
        
        # Critical issues
        critical_issues = []
        for gate in report["quality_gates"]:
            for metric in gate["metrics"]:
                if metric["severity"] == "critical" and not metric["passed"]:
                    critical_issues.append(f"{gate['gate_name']}: {metric['name']}")
        
        if critical_issues:
            print("🚨 Critical Issues:")
            for issue in critical_issues:
                print(f"   • {issue}")
            print()
        
        # Recommendations
        if report["recommendations"]:
            print("💡 Recommendations:")
            for rec in report["recommendations"][:5]:  # Show top 5
                print(f"   • {rec}")
            if len(report["recommendations"]) > 5:
                print(f"   ... and {len(report['recommendations']) - 5} more")
            print()
        
        print("="*80)
        
        return summary["overall_status"] == "PASSED"


def main():
    """Main function to run quality assurance checks."""
    generator = QualityReportGenerator()
    
    try:
        # Generate report
        report = generator.generate_report("quality_gates_report.json")
        
        # Print summary
        passed = generator.print_summary(report)
        
        # Exit with appropriate code
        exit_code = 0 if passed else 1
        
        if passed:
            print("🎉 All quality gates passed! Ready for production deployment.")
        else:
            print("⚠️  Some quality gates failed. Please review and address issues before deployment.")
        
        return exit_code
        
    except Exception as e:
        print(f"❌ Quality assurance check failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())