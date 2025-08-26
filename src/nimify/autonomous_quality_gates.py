"""Autonomous Quality Gates System for SDLC validation."""

import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import ast
import re


class QualityGate:
    """Base class for quality gates."""
    
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.results: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def run_check(self, check_name: str, check_function, *args, **kwargs) -> Dict[str, Any]:
        """Run a quality check with error handling."""
        start_time = time.time()
        
        try:
            result = check_function(*args, **kwargs)
            success = bool(result) if isinstance(result, bool) else result.get('success', False)
            
            check_result = {
                'check': check_name,
                'success': success,
                'result': result,
                'duration': time.time() - start_time,
                'timestamp': datetime.utcnow().isoformat(),
                'error': None
            }
            
        except Exception as e:
            check_result = {
                'check': check_name,
                'success': False,
                'result': None,
                'duration': time.time() - start_time,
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
            
            self.logger.error(f"Quality check '{check_name}' failed: {e}")
        
        self.results.append(check_result)
        return check_result
    
    def get_score(self) -> float:
        """Calculate quality gate score (0-100)."""
        if not self.results:
            return 0.0
        
        successful_checks = sum(1 for r in self.results if r['success'])
        return (successful_checks / len(self.results)) * 100


class CodeQualityGate(QualityGate):
    """Code quality validation gate."""
    
    def __init__(self):
        super().__init__("Code Quality", weight=1.5)
    
    def validate(self, project_path: Path) -> Dict[str, Any]:
        """Run comprehensive code quality validation."""
        self.logger.info("🔍 Running Code Quality Gates...")
        
        # Check 1: Python syntax validation
        self.run_check("syntax_validation", self._validate_syntax, project_path)
        
        # Check 2: Import validation
        self.run_check("import_validation", self._validate_imports, project_path)
        
        # Check 3: Code structure validation
        self.run_check("structure_validation", self._validate_structure, project_path)
        
        # Check 4: Documentation validation
        self.run_check("documentation_validation", self._validate_documentation, project_path)
        
        # Check 5: Code complexity validation
        self.run_check("complexity_validation", self._validate_complexity, project_path)
        
        score = self.get_score()
        return {
            'gate': self.name,
            'score': score,
            'passed': score >= 70.0,
            'checks': self.results,
            'recommendations': self._generate_recommendations()
        }
    
    def _validate_syntax(self, project_path: Path) -> Dict[str, Any]:
        """Validate Python syntax across all files."""
        python_files = list(project_path.rglob("*.py"))
        syntax_errors = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                    ast.parse(source)
            except SyntaxError as e:
                syntax_errors.append({
                    'file': str(py_file),
                    'error': str(e),
                    'line': e.lineno
                })
            except Exception as e:
                syntax_errors.append({
                    'file': str(py_file),
                    'error': f"Parse error: {str(e)}"
                })
        
        return {
            'success': len(syntax_errors) == 0,
            'files_checked': len(python_files),
            'syntax_errors': syntax_errors
        }
    
    def _validate_imports(self, project_path: Path) -> Dict[str, Any]:
        """Validate imports and dependencies."""
        python_files = list(project_path.rglob("*.py"))
        import_issues = []
        all_imports = set()
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Extract imports using regex
                import_pattern = r'^(?:from\s+(\S+)\s+)?import\s+(.+)$'
                for line_num, line in enumerate(content.split('\n'), 1):
                    line = line.strip()
                    if line.startswith(('import ', 'from ')):
                        match = re.match(import_pattern, line)
                        if match:
                            module = match.group(1) if match.group(1) else match.group(2).split(',')[0].strip()
                            all_imports.add(module)
                
            except Exception as e:
                import_issues.append({
                    'file': str(py_file),
                    'error': f"Import analysis error: {str(e)}"
                })
        
        # Check for common missing dependencies
        critical_imports = {'fastapi', 'uvicorn', 'pydantic', 'prometheus_client'}
        missing_critical = critical_imports - all_imports
        
        return {
            'success': len(import_issues) == 0 and len(missing_critical) == 0,
            'files_analyzed': len(python_files),
            'import_issues': import_issues,
            'total_imports': len(all_imports),
            'missing_critical_imports': list(missing_critical)
        }
    
    def _validate_structure(self, project_path: Path) -> Dict[str, Any]:
        """Validate code structure and organization."""
        required_files = [
            'src/nimify/__init__.py',
            'src/nimify/core.py',
            'src/nimify/api.py',
            'src/nimify/cli.py',
            'pyproject.toml'
        ]
        
        missing_files = []
        for required_file in required_files:
            if not (project_path / required_file).exists():
                missing_files.append(required_file)
        
        # Check for proper package structure
        nimify_path = project_path / 'src' / 'nimify'
        python_modules = len(list(nimify_path.glob('*.py'))) if nimify_path.exists() else 0
        
        return {
            'success': len(missing_files) == 0 and python_modules >= 5,
            'missing_files': missing_files,
            'python_modules_count': python_modules,
            'has_proper_structure': nimify_path.exists()
        }
    
    def _validate_documentation(self, project_path: Path) -> Dict[str, Any]:
        """Validate documentation coverage."""
        python_files = list((project_path / 'src').rglob("*.py")) if (project_path / 'src').exists() else []
        documented_functions = 0
        total_functions = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            total_functions += 1
                            if (ast.get_docstring(node) is not None):
                                documented_functions += 1
                                
            except Exception:
                continue
        
        documentation_coverage = (documented_functions / max(1, total_functions)) * 100
        
        return {
            'success': documentation_coverage >= 50.0,
            'documentation_coverage': documentation_coverage,
            'documented_items': documented_functions,
            'total_items': total_functions
        }
    
    def _validate_complexity(self, project_path: Path) -> Dict[str, Any]:
        """Validate code complexity metrics."""
        python_files = list((project_path / 'src').rglob("*.py")) if (project_path / 'src').exists() else []
        complex_functions = []
        total_functions = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            total_functions += 1
                            complexity = self._calculate_cyclomatic_complexity(node)
                            
                            if complexity > 10:  # High complexity threshold
                                complex_functions.append({
                                    'function': node.name,
                                    'file': str(py_file.relative_to(project_path)),
                                    'complexity': complexity,
                                    'line': node.lineno
                                })
                                
            except Exception:
                continue
        
        return {
            'success': len(complex_functions) < (total_functions * 0.1),  # Less than 10% complex
            'complex_functions': complex_functions,
            'total_functions': total_functions,
            'avg_complexity': 5.0  # Estimated average
        }
    
    def _calculate_cyclomatic_complexity(self, node) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.Try):
                complexity += len(child.handlers)
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _generate_recommendations(self) -> List[str]:
        """Generate code quality recommendations."""
        recommendations = []
        
        for result in self.results:
            if not result['success']:
                if result['check'] == 'syntax_validation':
                    recommendations.append("Fix Python syntax errors in identified files")
                elif result['check'] == 'import_validation':
                    recommendations.append("Install missing dependencies and fix import issues")
                elif result['check'] == 'structure_validation':
                    recommendations.append("Organize code into proper package structure")
                elif result['check'] == 'documentation_validation':
                    recommendations.append("Add docstrings to functions and classes")
                elif result['check'] == 'complexity_validation':
                    recommendations.append("Refactor complex functions to improve maintainability")
        
        return recommendations


class SecurityGate(QualityGate):
    """Security validation gate."""
    
    def __init__(self):
        super().__init__("Security", weight=2.0)
    
    def validate(self, project_path: Path) -> Dict[str, Any]:
        """Run comprehensive security validation."""
        self.logger.info("🔒 Running Security Gates...")
        
        # Check 1: Secrets scanning
        self.run_check("secrets_scan", self._scan_secrets, project_path)
        
        # Check 2: Security best practices
        self.run_check("security_practices", self._check_security_practices, project_path)
        
        # Check 3: Dependency vulnerabilities
        self.run_check("dependency_security", self._check_dependency_security, project_path)
        
        # Check 4: Input validation
        self.run_check("input_validation", self._check_input_validation, project_path)
        
        # Check 5: Security headers
        self.run_check("security_headers", self._check_security_headers, project_path)
        
        score = self.get_score()
        return {
            'gate': self.name,
            'score': score,
            'passed': score >= 80.0,
            'checks': self.results,
            'recommendations': self._generate_security_recommendations()
        }
    
    def _scan_secrets(self, project_path: Path) -> Dict[str, Any]:
        """Scan for potential secrets in code."""
        python_files = list(project_path.rglob("*.py"))
        config_files = list(project_path.rglob("*.yml")) + list(project_path.rglob("*.yaml")) + list(project_path.rglob("*.env"))
        
        secret_patterns = [
            (r'api[_-]?key\s*=\s*["\']([^"\']+)["\']', 'API Key'),
            (r'secret[_-]?key\s*=\s*["\']([^"\']+)["\']', 'Secret Key'),
            (r'password\s*=\s*["\']([^"\']+)["\']', 'Password'),
            (r'token\s*=\s*["\']([^"\']+)["\']', 'Token'),
            (r'["\'][A-Za-z0-9]{32,}["\']', 'Potential Secret'),
        ]
        
        potential_secrets = []
        
        for file_path in python_files + config_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    for pattern, secret_type in secret_patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            # Skip common false positives
                            matched_text = match.group(1) if match.groups() else match.group(0)
                            if matched_text.lower() in ['your-api-key', 'placeholder', 'example', 'test', 'dummy']:
                                continue
                                
                            potential_secrets.append({
                                'file': str(file_path.relative_to(project_path)),
                                'type': secret_type,
                                'line': content[:match.start()].count('\n') + 1
                            })
                            
            except Exception:
                continue
        
        return {
            'success': len(potential_secrets) == 0,
            'potential_secrets': potential_secrets,
            'files_scanned': len(python_files + config_files)
        }
    
    def _check_security_practices(self, project_path: Path) -> Dict[str, Any]:
        """Check for security best practices implementation."""
        practices = {
            'input_validation': False,
            'error_handling': False,
            'authentication': False,
            'authorization': False,
            'logging': False
        }
        
        python_files = list((project_path / 'src').rglob("*.py")) if (project_path / 'src').exists() else []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                    if 'pydantic' in content or 'validator' in content:
                        practices['input_validation'] = True
                    
                    if 'try:' in content and 'except' in content:
                        practices['error_handling'] = True
                    
                    if 'authentication' in content or 'auth' in content:
                        practices['authentication'] = True
                    
                    if 'authorization' in content or 'permission' in content:
                        practices['authorization'] = True
                    
                    if 'logging' in content or 'logger' in content:
                        practices['logging'] = True
                        
            except Exception:
                continue
        
        practices_count = sum(practices.values())
        
        return {
            'success': practices_count >= 3,  # At least 3 practices implemented
            'implemented_practices': practices,
            'practices_count': practices_count
        }
    
    def _check_dependency_security(self, project_path: Path) -> Dict[str, Any]:
        """Check for known security vulnerabilities in dependencies."""
        pyproject_path = project_path / 'pyproject.toml'
        requirements_path = project_path / 'requirements.txt'
        
        dependencies = []
        
        # Check pyproject.toml
        if pyproject_path.exists():
            try:
                with open(pyproject_path, 'r') as f:
                    content = f.read()
                    # Extract dependencies (simplified)
                    import_matches = re.findall(r'["\']([a-zA-Z0-9_-]+)>=?[^"\']*["\']', content)
                    dependencies.extend(import_matches)
            except Exception:
                pass
        
        # Known vulnerable packages (simplified check)
        vulnerable_packages = {'pillow': '8.0.0', 'requests': '2.25.0'}
        
        vulnerabilities = []
        for dep in dependencies:
            if dep.lower() in vulnerable_packages:
                vulnerabilities.append({
                    'package': dep,
                    'vulnerability': f'Known issues in versions before {vulnerable_packages[dep.lower()]}'
                })
        
        return {
            'success': len(vulnerabilities) == 0,
            'dependencies_checked': len(dependencies),
            'vulnerabilities': vulnerabilities
        }
    
    def _check_input_validation(self, project_path: Path) -> Dict[str, Any]:
        """Check for proper input validation implementation."""
        python_files = list((project_path / 'src').rglob("*.py")) if (project_path / 'src').exists() else []
        validation_patterns = 0
        total_functions = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            total_functions += 1
                            
                            # Check for validation patterns in function
                            func_source = ast.get_source_segment(content, node)
                            if func_source and any(pattern in func_source.lower() for pattern in 
                                                 ['validate', 'check', 'isinstance', 'raise']):
                                validation_patterns += 1
                                
            except Exception:
                continue
        
        validation_rate = (validation_patterns / max(1, total_functions)) * 100
        
        return {
            'success': validation_rate >= 30.0,  # At least 30% of functions have validation
            'validation_rate': validation_rate,
            'validated_functions': validation_patterns,
            'total_functions': total_functions
        }
    
    def _check_security_headers(self, project_path: Path) -> Dict[str, Any]:
        """Check for security headers implementation."""
        api_files = list((project_path / 'src').rglob("*api*.py"))
        security_headers = []
        
        security_patterns = [
            'CORSMiddleware',
            'TrustedHostMiddleware',
            'HTTPSRedirectMiddleware',
            'security',
            'csrf',
            'helmet'
        ]
        
        for api_file in api_files:
            try:
                with open(api_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    for pattern in security_patterns:
                        if pattern in content:
                            security_headers.append(pattern)
                            
            except Exception:
                continue
        
        return {
            'success': len(security_headers) >= 2,
            'security_headers_found': list(set(security_headers)),
            'files_checked': len(api_files)
        }
    
    def _generate_security_recommendations(self) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        for result in self.results:
            if not result['success']:
                if result['check'] == 'secrets_scan':
                    recommendations.append("Remove hardcoded secrets and use environment variables")
                elif result['check'] == 'security_practices':
                    recommendations.append("Implement more security best practices (auth, validation, etc.)")
                elif result['check'] == 'dependency_security':
                    recommendations.append("Update vulnerable dependencies to secure versions")
                elif result['check'] == 'input_validation':
                    recommendations.append("Add comprehensive input validation to all functions")
                elif result['check'] == 'security_headers':
                    recommendations.append("Implement security middleware and headers")
        
        return recommendations


class PerformanceGate(QualityGate):
    """Performance validation gate."""
    
    def __init__(self):
        super().__init__("Performance", weight=1.2)
    
    def validate(self, project_path: Path) -> Dict[str, Any]:
        """Run performance validation."""
        self.logger.info("⚡ Running Performance Gates...")
        
        # Check 1: Performance monitoring
        self.run_check("monitoring_implementation", self._check_monitoring, project_path)
        
        # Check 2: Caching implementation
        self.run_check("caching_implementation", self._check_caching, project_path)
        
        # Check 3: Async/await usage
        self.run_check("async_implementation", self._check_async_usage, project_path)
        
        # Check 4: Resource optimization
        self.run_check("resource_optimization", self._check_resource_optimization, project_path)
        
        # Check 5: Performance best practices
        self.run_check("performance_practices", self._check_performance_practices, project_path)
        
        score = self.get_score()
        return {
            'gate': self.name,
            'score': score,
            'passed': score >= 70.0,
            'checks': self.results,
            'recommendations': self._generate_performance_recommendations()
        }
    
    def _check_monitoring(self, project_path: Path) -> Dict[str, Any]:
        """Check for performance monitoring implementation."""
        python_files = list((project_path / 'src').rglob("*.py")) if (project_path / 'src').exists() else []
        monitoring_features = []
        
        monitoring_patterns = [
            'prometheus',
            'metrics',
            'counter',
            'histogram',
            'gauge',
            'monitoring',
            'telemetry'
        ]
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                    for pattern in monitoring_patterns:
                        if pattern in content:
                            monitoring_features.append(pattern)
                            
            except Exception:
                continue
        
        unique_features = list(set(monitoring_features))
        
        return {
            'success': len(unique_features) >= 3,
            'monitoring_features': unique_features,
            'files_checked': len(python_files)
        }
    
    def _check_caching(self, project_path: Path) -> Dict[str, Any]:
        """Check for caching implementation."""
        python_files = list((project_path / 'src').rglob("*.py")) if (project_path / 'src').exists() else []
        caching_patterns = ['cache', 'lru_cache', 'redis', 'memcached']
        caching_found = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                    for pattern in caching_patterns:
                        if pattern in content:
                            caching_found.append(pattern)
                            
            except Exception:
                continue
        
        return {
            'success': len(set(caching_found)) >= 1,
            'caching_implementations': list(set(caching_found)),
            'files_checked': len(python_files)
        }
    
    def _check_async_usage(self, project_path: Path) -> Dict[str, Any]:
        """Check for async/await usage."""
        python_files = list((project_path / 'src').rglob("*.py")) if (project_path / 'src').exists() else []
        async_functions = 0
        total_functions = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            total_functions += 1
                        elif isinstance(node, ast.AsyncFunctionDef):
                            async_functions += 1
                            total_functions += 1
                            
            except Exception:
                continue
        
        async_ratio = (async_functions / max(1, total_functions)) * 100
        
        return {
            'success': async_ratio >= 20.0,  # At least 20% async functions
            'async_ratio': async_ratio,
            'async_functions': async_functions,
            'total_functions': total_functions
        }
    
    def _check_resource_optimization(self, project_path: Path) -> Dict[str, Any]:
        """Check for resource optimization patterns."""
        python_files = list((project_path / 'src').rglob("*.py")) if (project_path / 'src').exists() else []
        optimization_patterns = [
            'connection_pool',
            'threadpool',
            'asyncio',
            'concurrent',
            'multiprocessing',
            'generator',
            'lazy'
        ]
        
        optimizations_found = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                    for pattern in optimization_patterns:
                        if pattern in content:
                            optimizations_found.append(pattern)
                            
            except Exception:
                continue
        
        return {
            'success': len(set(optimizations_found)) >= 2,
            'optimizations_found': list(set(optimizations_found)),
            'files_checked': len(python_files)
        }
    
    def _check_performance_practices(self, project_path: Path) -> Dict[str, Any]:
        """Check for performance best practices."""
        practices = {
            'batch_processing': False,
            'streaming': False,
            'pagination': False,
            'compression': False,
            'indexing': False
        }
        
        python_files = list((project_path / 'src').rglob("*.py")) if (project_path / 'src').exists() else []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                    if 'batch' in content:
                        practices['batch_processing'] = True
                    if 'stream' in content:
                        practices['streaming'] = True
                    if 'page' in content or 'limit' in content:
                        practices['pagination'] = True
                    if 'gzip' in content or 'compress' in content:
                        practices['compression'] = True
                    if 'index' in content:
                        practices['indexing'] = True
                        
            except Exception:
                continue
        
        practices_count = sum(practices.values())
        
        return {
            'success': practices_count >= 2,
            'practices_implemented': practices,
            'practices_count': practices_count
        }
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        for result in self.results:
            if not result['success']:
                if result['check'] == 'monitoring_implementation':
                    recommendations.append("Implement comprehensive performance monitoring with Prometheus")
                elif result['check'] == 'caching_implementation':
                    recommendations.append("Add caching layers to improve response times")
                elif result['check'] == 'async_implementation':
                    recommendations.append("Convert I/O-bound functions to async/await pattern")
                elif result['check'] == 'resource_optimization':
                    recommendations.append("Implement connection pooling and resource optimization")
                elif result['check'] == 'performance_practices':
                    recommendations.append("Implement batch processing, streaming, and compression")
        
        return recommendations


class AutonomousQualityGates:
    """Autonomous Quality Gates System."""
    
    def __init__(self, project_path: Optional[Path] = None):
        self.project_path = project_path or Path.cwd()
        self.logger = logging.getLogger(__name__)
        
        # Initialize quality gates
        self.gates = [
            CodeQualityGate(),
            SecurityGate(),
            PerformanceGate()
        ]
        
        self.results: Dict[str, Any] = {}
        self.overall_score = 0.0
        self.passed = False
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates autonomously."""
        self.logger.info("🚀 Running Autonomous Quality Gates...")
        
        gate_results = []
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for gate in self.gates:
            try:
                result = gate.validate(self.project_path)
                gate_results.append(result)
                
                # Calculate weighted score
                total_weighted_score += result['score'] * gate.weight
                total_weight += gate.weight
                
                status = "✅" if result['passed'] else "❌"
                self.logger.info(f"{status} {result['gate']}: {result['score']:.1f}%")
                
            except Exception as e:
                self.logger.error(f"Gate {gate.name} failed with error: {e}")
                gate_results.append({
                    'gate': gate.name,
                    'score': 0.0,
                    'passed': False,
                    'error': str(e),
                    'checks': [],
                    'recommendations': [f"Fix gate execution error: {str(e)}"]
                })
        
        # Calculate overall score
        self.overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        self.passed = self.overall_score >= 75.0
        
        # Generate comprehensive results
        self.results = {
            'timestamp': datetime.utcnow().isoformat(),
            'project_path': str(self.project_path),
            'overall_score': self.overall_score,
            'passed': self.passed,
            'gate_results': gate_results,
            'summary': self._generate_summary(gate_results),
            'recommendations': self._consolidate_recommendations(gate_results),
            'next_steps': self._generate_next_steps(gate_results)
        }
        
        return self.results
    
    def _generate_summary(self, gate_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate quality gates summary."""
        passed_gates = sum(1 for result in gate_results if result['passed'])
        total_gates = len(gate_results)
        
        gate_scores = {result['gate']: result['score'] for result in gate_results}
        
        return {
            'gates_passed': f"{passed_gates}/{total_gates}",
            'gate_scores': gate_scores,
            'weakest_area': min(gate_scores, key=gate_scores.get) if gate_scores else None,
            'strongest_area': max(gate_scores, key=gate_scores.get) if gate_scores else None,
            'overall_status': "PASSED" if self.passed else "NEEDS_IMPROVEMENT"
        }
    
    def _consolidate_recommendations(self, gate_results: List[Dict[str, Any]]) -> List[str]:
        """Consolidate recommendations from all gates."""
        all_recommendations = []
        
        for result in gate_results:
            recommendations = result.get('recommendations', [])
            all_recommendations.extend(recommendations)
        
        # Remove duplicates while preserving order
        unique_recommendations = []
        seen = set()
        
        for rec in all_recommendations:
            if rec not in seen:
                unique_recommendations.append(rec)
                seen.add(rec)
        
        return unique_recommendations[:15]  # Top 15 recommendations
    
    def _generate_next_steps(self, gate_results: List[Dict[str, Any]]) -> List[str]:
        """Generate prioritized next steps."""
        next_steps = []
        
        # Prioritize by gate scores (lowest first)
        sorted_gates = sorted(gate_results, key=lambda x: x['score'])
        
        for result in sorted_gates[:3]:  # Top 3 priority areas
            gate_name = result['gate']
            score = result['score']
            
            if score < 50:
                priority = "🔴 CRITICAL"
            elif score < 75:
                priority = "🟡 HIGH"
            else:
                priority = "🟢 LOW"
            
            next_steps.append(f"{priority}: Improve {gate_name} (current: {score:.1f}%)")
        
        return next_steps
    
    def save_results(self, output_path: Optional[Path] = None) -> Path:
        """Save results to JSON file."""
        if not output_path:
            output_path = self.project_path / "autonomous_quality_gates_report.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📄 Quality gates report saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print a summary of the quality gates results."""
        print("\n" + "="*60)
        print("🏁 AUTONOMOUS QUALITY GATES RESULTS")
        print("="*60)
        
        print(f"\n📊 Overall Score: {self.overall_score:.1f}%")
        print(f"🎯 Status: {'✅ PASSED' if self.passed else '❌ NEEDS IMPROVEMENT'}")
        
        if 'gate_results' in self.results:
            print(f"\n🔍 Gate Results:")
            for result in self.results['gate_results']:
                status = "✅" if result['passed'] else "❌"
                print(f"   {status} {result['gate']}: {result['score']:.1f}%")
        
        if 'recommendations' in self.results and self.results['recommendations']:
            print(f"\n💡 Top Recommendations:")
            for i, rec in enumerate(self.results['recommendations'][:5], 1):
                print(f"   {i}. {rec}")
        
        if 'next_steps' in self.results and self.results['next_steps']:
            print(f"\n🚀 Next Steps:")
            for step in self.results['next_steps']:
                print(f"   • {step}")
        
        print("\n" + "="*60)


def main():
    """Main function to run autonomous quality gates."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    quality_gates = AutonomousQualityGates()
    results = quality_gates.run_all_gates()
    
    # Save results
    quality_gates.save_results()
    
    # Print summary
    quality_gates.print_summary()
    
    # Return exit code based on results
    return 0 if results['passed'] else 1


if __name__ == "__main__":
    exit(main())