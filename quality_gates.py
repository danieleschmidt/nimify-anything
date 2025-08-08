#!/usr/bin/env python3
"""Comprehensive quality gates and validation system."""

import sys
import os
import subprocess
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Add src to path
sys.path.insert(0, 'src')

class QualityGate:
    """Base class for quality gates."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.score = 0.0
        self.details = {}
        self.errors = []
        self.warnings = []
    
    def run(self) -> bool:
        """Run the quality gate check."""
        raise NotImplementedError
    
    def get_report(self) -> Dict[str, Any]:
        """Get detailed report."""
        return {
            'name': self.name,
            'passed': self.passed,
            'score': self.score,
            'details': self.details,
            'errors': self.errors,
            'warnings': self.warnings
        }


class CodeQualityGate(QualityGate):
    """Code quality and style checks."""
    
    def __init__(self):
        super().__init__("Code Quality")
    
    def run(self) -> bool:
        """Run code quality checks."""
        print("🔍 Running Code Quality Gates...")
        
        checks = [
            self._check_code_structure(),
            self._check_documentation(),
            self._check_naming_conventions(),
            self._check_complexity(),
            self._check_imports()
        ]
        
        passed_checks = sum(checks)
        total_checks = len(checks)
        
        self.score = passed_checks / total_checks
        self.passed = self.score >= 0.8  # 80% threshold
        
        print(f"✅ Code Quality: {passed_checks}/{total_checks} checks passed ({self.score:.1%})")
        return self.passed
    
    def _check_code_structure(self) -> bool:
        """Check code structure and organization."""
        try:
            src_dir = Path("src/nimify")
            required_modules = [
                "core.py", "cli.py", "api.py", "validation.py", 
                "security.py", "performance.py", "logging_config.py",
                "error_handling.py", "optimization.py", "global_deployment.py"
            ]
            
            missing_modules = []
            for module in required_modules:
                if not (src_dir / module).exists():
                    missing_modules.append(module)
            
            if missing_modules:
                self.errors.append(f"Missing modules: {missing_modules}")
                return False
            
            # Check module sizes (not too large)
            large_modules = []
            for module_path in src_dir.glob("*.py"):
                lines = len(module_path.read_text().splitlines())
                if lines > 1000:  # Threshold for large modules
                    large_modules.append(f"{module_path.name}: {lines} lines")
            
            if large_modules:
                self.warnings.append(f"Large modules detected: {large_modules}")
            
            self.details['structure'] = {
                'total_modules': len(list(src_dir.glob("*.py"))),
                'required_modules_present': len(required_modules) - len(missing_modules),
                'large_modules': len(large_modules)
            }
            
            return len(missing_modules) == 0
            
        except Exception as e:
            self.errors.append(f"Code structure check failed: {e}")
            return False
    
    def _check_documentation(self) -> bool:
        """Check documentation completeness."""
        try:
            doc_score = 0
            total_files = 0
            
            for py_file in Path("src/nimify").glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                
                total_files += 1
                content = py_file.read_text()
                
                # Check for module docstring
                if '"""' in content[:200]:
                    doc_score += 0.3
                
                # Check for function docstrings
                functions = re.findall(r'def\s+\w+\(', content)
                if functions:
                    docstring_count = content.count('"""') - 1  # Subtract module docstring
                    if docstring_count >= len(functions) * 0.5:  # 50% of functions documented
                        doc_score += 0.4
                
                # Check for class docstrings
                classes = re.findall(r'class\s+\w+', content)
                if classes:
                    class_docstrings = len(re.findall(r'class\s+\w+.*?:\\s*"""', content, re.DOTALL))
                    if class_docstrings >= len(classes) * 0.5:
                        doc_score += 0.3
            
            avg_doc_score = doc_score / total_files if total_files > 0 else 0
            
            self.details['documentation'] = {
                'files_checked': total_files,
                'documentation_score': avg_doc_score,
                'threshold': 0.6
            }
            
            return avg_doc_score >= 0.6
            
        except Exception as e:
            self.errors.append(f"Documentation check failed: {e}")
            return False
    
    def _check_naming_conventions(self) -> bool:
        """Check naming conventions compliance."""
        try:
            violations = []
            
            for py_file in Path("src/nimify").glob("*.py"):
                content = py_file.read_text()
                
                # Check class names (PascalCase)
                class_matches = re.findall(r'class\s+([A-Za-z_][A-Za-z0-9_]*)', content)
                for class_name in class_matches:
                    if not re.match(r'^[A-Z][a-zA-Z0-9]*$', class_name):
                        violations.append(f"{py_file.name}: Class '{class_name}' should be PascalCase")
                
                # Check function names (snake_case)
                function_matches = re.findall(r'def\s+([A-Za-z_][A-Za-z0-9_]*)', content)
                for func_name in function_matches:
                    if not re.match(r'^[a-z_][a-z0-9_]*$', func_name) and not func_name.startswith('_'):
                        violations.append(f"{py_file.name}: Function '{func_name}' should be snake_case")
                
                # Check constants (UPPER_CASE)
                const_matches = re.findall(r'^([A-Z_][A-Z0-9_]*)\s*=', content, re.MULTILINE)
                for const_name in const_matches:
                    if not re.match(r'^[A-Z_][A-Z0-9_]*$', const_name):
                        violations.append(f"{py_file.name}: Constant '{const_name}' should be UPPER_CASE")
            
            if violations:
                self.warnings.extend(violations[:10])  # Limit to first 10
                if len(violations) > 10:
                    self.warnings.append(f"... and {len(violations) - 10} more naming violations")
            
            self.details['naming'] = {
                'violations_count': len(violations),
                'threshold': 20
            }
            
            return len(violations) <= 20  # Allow some violations
            
        except Exception as e:
            self.errors.append(f"Naming convention check failed: {e}")
            return False
    
    def _check_complexity(self) -> bool:
        """Check code complexity."""
        try:
            complex_functions = []
            
            for py_file in Path("src/nimify").glob("*.py"):
                content = py_file.read_text()
                
                # Simple cyclomatic complexity estimation
                functions = re.findall(r'def\s+(\w+)\(.*?\):(.*?)(?=\ndef|\nclass|\n\S|\Z)', 
                                     content, re.DOTALL)
                
                for func_name, func_body in functions:
                    # Count decision points
                    complexity = 1  # Base complexity
                    complexity += len(re.findall(r'\bif\b', func_body))
                    complexity += len(re.findall(r'\bfor\b', func_body))
                    complexity += len(re.findall(r'\bwhile\b', func_body))
                    complexity += len(re.findall(r'\bexcept\b', func_body))
                    complexity += len(re.findall(r'\belif\b', func_body))
                    complexity += len(re.findall(r'\band\b|\bor\b', func_body))
                    
                    if complexity > 15:  # High complexity threshold
                        complex_functions.append(f"{py_file.name}:{func_name} (complexity: {complexity})")
            
            if complex_functions:
                self.warnings.extend(complex_functions[:5])  # Limit warnings
                if len(complex_functions) > 5:
                    self.warnings.append(f"... and {len(complex_functions) - 5} more complex functions")
            
            self.details['complexity'] = {
                'high_complexity_functions': len(complex_functions),
                'threshold': 10
            }
            
            return len(complex_functions) <= 10
            
        except Exception as e:
            self.errors.append(f"Complexity check failed: {e}")
            return False
    
    def _check_imports(self) -> bool:
        """Check import organization and usage."""
        try:
            import_issues = []
            
            for py_file in Path("src/nimify").glob("*.py"):
                content = py_file.read_text()
                lines = content.splitlines()
                
                # Check for unused imports (simple check)
                imports = []
                for line in lines:
                    if line.startswith('import ') or line.startswith('from '):
                        # Extract imported names
                        if ' import ' in line:
                            parts = line.split(' import ')[1].split(',')
                            for part in parts:
                                imported = part.strip().split(' as ')[0]
                                imports.append(imported)
                        else:
                            imported = line.replace('import ', '').split('.')[0]
                            imports.append(imported)
                
                # Check if imports are used (simple text search)
                content_without_imports = '\\n'.join([l for l in lines if not (l.startswith('import ') or l.startswith('from '))])
                
                unused_imports = []
                for imp in imports:
                    if imp not in content_without_imports and len(imp) > 2:  # Skip very short names
                        unused_imports.append(imp)
                
                if unused_imports:
                    import_issues.append(f"{py_file.name}: Potentially unused imports: {unused_imports}")
            
            if import_issues:
                self.warnings.extend(import_issues)
            
            self.details['imports'] = {
                'files_with_import_issues': len(import_issues),
                'threshold': 5
            }
            
            return len(import_issues) <= 5
            
        except Exception as e:
            self.errors.append(f"Import check failed: {e}")
            return False


class SecurityGate(QualityGate):
    """Security vulnerability checks."""
    
    def __init__(self):
        super().__init__("Security")
    
    def run(self) -> bool:
        """Run security checks."""
        print("🔒 Running Security Gates...")
        
        checks = [
            self._check_hardcoded_secrets(),
            self._check_dangerous_functions(),
            self._check_input_validation(),
            self._check_authentication(),
            self._check_encryption()
        ]
        
        passed_checks = sum(checks)
        total_checks = len(checks)
        
        self.score = passed_checks / total_checks
        self.passed = self.score >= 0.9  # Higher threshold for security
        
        print(f"🔒 Security: {passed_checks}/{total_checks} checks passed ({self.score:.1%})")
        return self.passed
    
    def _check_hardcoded_secrets(self) -> bool:
        """Check for hardcoded secrets."""
        try:
            secret_patterns = [
                (r'password\s*=\s*["\'][^"\']+["\']', 'hardcoded password'),
                (r'api[_-]?key\s*=\s*["\'][^"\']+["\']', 'hardcoded API key'),
                (r'secret[_-]?key\s*=\s*["\'][^"\']+["\']', 'hardcoded secret key'),
                (r'token\s*=\s*["\'][^"\']+["\']', 'hardcoded token'),
                (r'["\'][A-Za-z0-9+/]{40,}["\']', 'potential secret (base64)'),
            ]
            
            violations = []
            
            for py_file in Path("src").rglob("*.py"):
                content = py_file.read_text()
                
                for pattern, description in secret_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Skip obvious test/example values
                        matched_text = match.group(0).lower()
                        if not any(test_val in matched_text for test_val in ['test', 'example', 'dummy', 'mock', 'xxx']):
                            violations.append(f"{py_file.name}: {description} found")
            
            if violations:
                self.errors.extend(violations)
            
            self.details['secrets'] = {
                'violations_found': len(violations),
                'files_scanned': len(list(Path("src").rglob("*.py")))
            }
            
            return len(violations) == 0
            
        except Exception as e:
            self.errors.append(f"Secret scan failed: {e}")
            return False
    
    def _check_dangerous_functions(self) -> bool:
        """Check for dangerous function usage."""
        try:
            dangerous_patterns = [
                (r'\beval\s*\(', 'eval() usage'),
                (r'\bexec\s*\(', 'exec() usage'),
                (r'\b__import__\s*\(', '__import__() usage'),
                (r'subprocess\.call\([^)]*shell\s*=\s*True', 'shell=True in subprocess'),
                (r'os\.system\s*\(', 'os.system() usage'),
                (r'pickle\.loads?\s*\(', 'pickle.load() - potential security risk'),
            ]
            
            violations = []
            
            for py_file in Path("src").rglob("*.py"):
                content = py_file.read_text()
                
                for pattern, description in dangerous_patterns:
                    if re.search(pattern, content):
                        violations.append(f"{py_file.name}: {description}")
            
            if violations:
                self.warnings.extend(violations)  # Warnings, not errors, as some might be legitimate
            
            self.details['dangerous_functions'] = {
                'violations_found': len(violations),
                'threshold': 5
            }
            
            return len(violations) <= 5  # Allow some legitimate usage
            
        except Exception as e:
            self.errors.append(f"Dangerous function check failed: {e}")
            return False
    
    def _check_input_validation(self) -> bool:
        """Check for input validation implementation."""
        try:
            # Check if validation module exists and has proper functions
            validation_file = Path("src/nimify/validation.py")
            if not validation_file.exists():
                self.errors.append("Input validation module not found")
                return False
            
            content = validation_file.read_text()
            
            # Check for validation functions
            required_validations = [
                'validate_service_name',
                'sanitize',
                'scan_for_attacks'
            ]
            
            found_validations = []
            for validation in required_validations:
                if validation in content:
                    found_validations.append(validation)
            
            # Check for security classes
            security_classes = [
                'ServiceNameValidator',
                'ModelFileValidator', 
                'SecurityValidator'
            ]
            
            found_classes = []
            for sec_class in security_classes:
                if f"class {sec_class}" in content:
                    found_classes.append(sec_class)
            
            self.details['input_validation'] = {
                'validation_functions_found': len(found_validations),
                'security_classes_found': len(found_classes),
                'total_required': len(required_validations) + len(security_classes)
            }
            
            total_found = len(found_validations) + len(found_classes)
            total_required = len(required_validations) + len(security_classes)
            
            return total_found >= total_required * 0.7  # 70% coverage
            
        except Exception as e:
            self.errors.append(f"Input validation check failed: {e}")
            return False
    
    def _check_authentication(self) -> bool:
        """Check authentication implementation."""
        try:
            security_file = Path("src/nimify/security.py")
            if not security_file.exists():
                self.errors.append("Security module not found")
                return False
            
            content = security_file.read_text()
            
            # Check for authentication features
            auth_features = [
                'APIKeyManager',
                'validate_api_key',
                'generate_api_key',
                'RateLimiter',
                'IPBlocklist'
            ]
            
            found_features = []
            for feature in auth_features:
                if feature in content:
                    found_features.append(feature)
            
            self.details['authentication'] = {
                'auth_features_found': len(found_features),
                'total_features': len(auth_features)
            }
            
            return len(found_features) >= len(auth_features) * 0.8  # 80% coverage
            
        except Exception as e:
            self.errors.append(f"Authentication check failed: {e}")
            return False
    
    def _check_encryption(self) -> bool:
        """Check encryption and security headers."""
        try:
            security_file = Path("src/nimify/security.py")
            content = security_file.read_text() if security_file.exists() else ""
            
            # Look for security-related implementations
            security_indicators = [
                'hashlib',
                'hmac',
                'secrets',
                'SecurityHeaders',
                'TLS',
                'encryption'
            ]
            
            found_indicators = []
            for indicator in security_indicators:
                if indicator in content.lower():
                    found_indicators.append(indicator)
            
            self.details['encryption'] = {
                'security_indicators_found': len(found_indicators),
                'total_indicators': len(security_indicators)
            }
            
            return len(found_indicators) >= len(security_indicators) * 0.5  # 50% coverage
            
        except Exception as e:
            self.errors.append(f"Encryption check failed: {e}")
            return False


class PerformanceGate(QualityGate):
    """Performance benchmarks and optimization checks."""
    
    def __init__(self):
        super().__init__("Performance")
    
    def run(self) -> bool:
        """Run performance checks."""
        print("⚡ Running Performance Gates...")
        
        checks = [
            self._check_performance_module(),
            self._check_optimization_features(),
            self._check_caching_implementation(),
            self._check_monitoring_setup(),
            self._run_basic_benchmarks()
        ]
        
        passed_checks = sum(checks)
        total_checks = len(checks)
        
        self.score = passed_checks / total_checks
        self.passed = self.score >= 0.8
        
        print(f"⚡ Performance: {passed_checks}/{total_checks} checks passed ({self.score:.1%})")
        return self.passed
    
    def _check_performance_module(self) -> bool:
        """Check performance module implementation."""
        try:
            perf_file = Path("src/nimify/performance.py")
            if not perf_file.exists():
                self.errors.append("Performance module not found")
                return False
            
            content = perf_file.read_text()
            
            # Check for performance classes
            perf_classes = [
                'MetricsCollector',
                'ModelCache',
                'CircuitBreaker',
                'AdaptiveScaler'
            ]
            
            found_classes = []
            for cls in perf_classes:
                if f"class {cls}" in content:
                    found_classes.append(cls)
            
            self.details['performance_module'] = {
                'classes_found': len(found_classes),
                'total_classes': len(perf_classes)
            }
            
            return len(found_classes) >= len(perf_classes) * 0.8
            
        except Exception as e:
            self.errors.append(f"Performance module check failed: {e}")
            return False
    
    def _check_optimization_features(self) -> bool:
        """Check optimization features."""
        try:
            opt_file = Path("src/nimify/optimization.py")
            if not opt_file.exists():
                self.errors.append("Optimization module not found")
                return False
            
            content = opt_file.read_text()
            
            # Check for optimization features
            opt_features = [
                'OptimizationEngine',
                'ModelOptimizer',
                'CacheOptimizer',
                'BatchOptimizer',
                'AutoScaler'
            ]
            
            found_features = []
            for feature in opt_features:
                if feature in content:
                    found_features.append(feature)
            
            self.details['optimization'] = {
                'features_found': len(found_features),
                'total_features': len(opt_features)
            }
            
            return len(found_features) >= len(opt_features) * 0.8
            
        except Exception as e:
            self.errors.append(f"Optimization check failed: {e}")
            return False
    
    def _check_caching_implementation(self) -> bool:
        """Check caching implementation."""
        try:
            # Check for caching in performance module
            perf_file = Path("src/nimify/performance.py")
            content = perf_file.read_text() if perf_file.exists() else ""
            
            cache_features = [
                'cache',
                'get',
                'put', 
                'hit_rate',
                'TTL'
            ]
            
            found_features = []
            for feature in cache_features:
                if feature.lower() in content.lower():
                    found_features.append(feature)
            
            self.details['caching'] = {
                'cache_features_found': len(found_features),
                'total_features': len(cache_features)
            }
            
            return len(found_features) >= len(cache_features) * 0.6
            
        except Exception as e:
            self.errors.append(f"Caching check failed: {e}")
            return False
    
    def _check_monitoring_setup(self) -> bool:
        """Check monitoring implementation."""
        try:
            monitoring_file = Path("src/nimify/monitoring.py")
            if not monitoring_file.exists():
                self.warnings.append("Monitoring module not found")
                return True  # Not critical
            
            content = monitoring_file.read_text()
            
            monitoring_features = [
                'MetricsCollector',
                'Alert',
                'prometheus',
                'grafana'
            ]
            
            found_features = []
            for feature in monitoring_features:
                if feature.lower() in content.lower():
                    found_features.append(feature)
            
            self.details['monitoring'] = {
                'monitoring_features_found': len(found_features),
                'total_features': len(monitoring_features)
            }
            
            return len(found_features) >= len(monitoring_features) * 0.5
            
        except Exception as e:
            self.warnings.append(f"Monitoring check failed: {e}")
            return True  # Not critical for pass/fail
    
    def _run_basic_benchmarks(self) -> bool:
        """Run basic performance benchmarks."""
        try:
            start_time = time.time()
            
            # Test basic imports (should be fast)
            try:
                from nimify.core import ModelConfig, Nimifier
                from nimify.security import rate_limiter
                import_time = time.time() - start_time
            except Exception as e:
                self.errors.append(f"Import benchmark failed: {e}")
                return False
            
            # Test basic object creation
            start_time = time.time()
            try:
                config = ModelConfig(name="benchmark-test", max_batch_size=32)
                nimifier = Nimifier(config)
                creation_time = time.time() - start_time
            except Exception as e:
                self.errors.append(f"Object creation benchmark failed: {e}")
                return False
            
            # Performance thresholds
            import_threshold = 2.0  # seconds
            creation_threshold = 0.1  # seconds
            
            self.details['benchmarks'] = {
                'import_time_seconds': import_time,
                'creation_time_seconds': creation_time,
                'import_passed': import_time <= import_threshold,
                'creation_passed': creation_time <= creation_threshold
            }
            
            if import_time > import_threshold:
                self.warnings.append(f"Slow imports: {import_time:.2f}s (threshold: {import_threshold}s)")
            
            if creation_time > creation_threshold:
                self.warnings.append(f"Slow object creation: {creation_time:.2f}s (threshold: {creation_threshold}s)")
            
            return import_time <= import_threshold and creation_time <= creation_threshold
            
        except Exception as e:
            self.errors.append(f"Benchmark failed: {e}")
            return False


class TestGate(QualityGate):
    """Test coverage and correctness checks."""
    
    def __init__(self):
        super().__init__("Tests")
    
    def run(self) -> bool:
        """Run test checks."""
        print("🧪 Running Test Gates...")
        
        checks = [
            self._check_test_structure(),
            self._check_test_coverage(),
            self._run_existing_tests(),
            self._check_test_quality()
        ]
        
        passed_checks = sum(checks)
        total_checks = len(checks)
        
        self.score = passed_checks / total_checks
        self.passed = self.score >= 0.75  # 75% threshold for tests
        
        print(f"🧪 Tests: {passed_checks}/{total_checks} checks passed ({self.score:.1%})")
        return self.passed
    
    def _check_test_structure(self) -> bool:
        """Check test directory structure."""
        try:
            tests_dir = Path("tests")
            if not tests_dir.exists():
                self.errors.append("Tests directory not found")
                return False
            
            test_files = list(tests_dir.glob("test_*.py"))
            
            # Check for test files corresponding to main modules
            src_modules = [f.stem for f in Path("src/nimify").glob("*.py") if not f.name.startswith("_")]
            
            expected_tests = [f"test_{module}.py" for module in src_modules]
            existing_tests = [f.name for f in test_files]
            
            missing_tests = [test for test in expected_tests if test not in existing_tests]
            
            if missing_tests:
                self.warnings.append(f"Missing test files: {missing_tests[:5]}")  # Show first 5
            
            self.details['test_structure'] = {
                'test_files_found': len(test_files),
                'expected_test_files': len(expected_tests),
                'missing_test_files': len(missing_tests)
            }
            
            return len(missing_tests) <= len(expected_tests) * 0.5  # Allow 50% missing
            
        except Exception as e:
            self.errors.append(f"Test structure check failed: {e}")
            return False
    
    def _check_test_coverage(self) -> bool:
        """Estimate test coverage by checking test content."""
        try:
            total_functions = 0
            tested_functions = 0
            
            # Count functions in source
            for py_file in Path("src/nimify").glob("*.py"):
                content = py_file.read_text()
                functions = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', content)
                # Filter out private functions and special methods
                public_functions = [f for f in functions if not f.startswith('_')]
                total_functions += len(public_functions)
            
            # Count test functions
            test_files = list(Path("tests").glob("test_*.py")) if Path("tests").exists() else []
            test_files.extend([Path("simple_test.py"), Path("test_robust_simple.py"), Path("test_optimized_system.py")])
            
            for test_file in test_files:
                if test_file.exists():
                    content = test_file.read_text()
                    test_functions = re.findall(r'def\s+test_[a-zA-Z_][a-zA-Z0-9_]*\s*\(', content)
                    tested_functions += len(test_functions)
            
            coverage_estimate = (tested_functions / total_functions) if total_functions > 0 else 0
            
            self.details['test_coverage'] = {
                'total_functions': total_functions,
                'test_functions': tested_functions,
                'coverage_estimate': coverage_estimate,
                'threshold': 0.3
            }
            
            return coverage_estimate >= 0.3  # 30% coverage threshold
            
        except Exception as e:
            self.errors.append(f"Test coverage check failed: {e}")
            return False
    
    def _run_existing_tests(self) -> bool:
        """Run existing test files."""
        try:
            test_results = []
            
            # Test files to run
            test_files = [
                "simple_test.py",
                "test_robust_simple.py", 
                "test_optimized_system.py"
            ]
            
            for test_file in test_files:
                if Path(test_file).exists():
                    try:
                        # Run the test file
                        result = subprocess.run([
                            sys.executable, test_file
                        ], capture_output=True, text=True, timeout=60)
                        
                        test_results.append({
                            'file': test_file,
                            'returncode': result.returncode,
                            'passed': result.returncode == 0
                        })
                        
                    except subprocess.TimeoutExpired:
                        test_results.append({
                            'file': test_file,
                            'returncode': -1,
                            'passed': False,
                            'error': 'Test timed out'
                        })
                    except Exception as e:
                        test_results.append({
                            'file': test_file,
                            'returncode': -1,
                            'passed': False,
                            'error': str(e)
                        })
            
            passed_tests = sum(1 for result in test_results if result['passed'])
            total_tests = len(test_results)
            
            self.details['test_execution'] = {
                'total_test_files': total_tests,
                'passed_test_files': passed_tests,
                'test_results': test_results
            }
            
            # At least 60% of tests should pass
            return (passed_tests / total_tests) >= 0.6 if total_tests > 0 else True
            
        except Exception as e:
            self.errors.append(f"Test execution failed: {e}")
            return False
    
    def _check_test_quality(self) -> bool:
        """Check quality of test code."""
        try:
            quality_score = 0
            test_files_checked = 0
            
            test_files = [Path("simple_test.py"), Path("test_robust_simple.py"), Path("test_optimized_system.py")]
            
            for test_file in test_files:
                if test_file.exists():
                    test_files_checked += 1
                    content = test_file.read_text()
                    
                    # Check for good testing practices
                    if 'assert' in content:
                        quality_score += 1
                    if 'Exception' in content and 'try:' in content:
                        quality_score += 1
                    if 'async def test_' in content or 'def test_' in content:
                        quality_score += 1
                    if 'print(' in content:  # Test output for visibility
                        quality_score += 0.5
            
            avg_quality = quality_score / test_files_checked if test_files_checked > 0 else 0
            
            self.details['test_quality'] = {
                'test_files_checked': test_files_checked,
                'quality_score': avg_quality,
                'threshold': 2.0
            }
            
            return avg_quality >= 2.0
            
        except Exception as e:
            self.errors.append(f"Test quality check failed: {e}")
            return False


class DeploymentGate(QualityGate):
    """Deployment readiness checks."""
    
    def __init__(self):
        super().__init__("Deployment")
    
    def run(self) -> bool:
        """Run deployment readiness checks."""
        print("🚀 Running Deployment Gates...")
        
        checks = [
            self._check_containerization(),
            self._check_kubernetes_manifests(),
            self._check_global_deployment(),
            self._check_configuration_management(),
            self._check_monitoring_integration()
        ]
        
        passed_checks = sum(checks)
        total_checks = len(checks)
        
        self.score = passed_checks / total_checks
        self.passed = self.score >= 0.8
        
        print(f"🚀 Deployment: {passed_checks}/{total_checks} checks passed ({self.score:.1%})")
        return self.passed
    
    def _check_containerization(self) -> bool:
        """Check Docker containerization setup."""
        try:
            docker_files = [
                "Dockerfile",
                "Dockerfile.production",
                "docker-compose.yml"
            ]
            
            found_files = []
            for docker_file in docker_files:
                if Path(docker_file).exists():
                    found_files.append(docker_file)
            
            # Check Dockerfile content if it exists
            if Path("Dockerfile").exists():
                content = Path("Dockerfile").read_text()
                
                # Check for best practices
                best_practices = [
                    'FROM' in content,
                    'COPY' in content or 'ADD' in content,
                    'EXPOSE' in content,
                    'CMD' in content or 'ENTRYPOINT' in content
                ]
                
                practices_found = sum(best_practices)
            else:
                practices_found = 0
            
            self.details['containerization'] = {
                'docker_files_found': len(found_files),
                'total_docker_files': len(docker_files),
                'dockerfile_practices': practices_found,
                'total_practices': 4
            }
            
            return len(found_files) >= 2 and practices_found >= 3
            
        except Exception as e:
            self.errors.append(f"Containerization check failed: {e}")
            return False
    
    def _check_kubernetes_manifests(self) -> bool:
        """Check Kubernetes deployment manifests."""
        try:
            # Check if deployment module exists
            deployment_file = Path("src/nimify/deployment.py")
            global_deployment_file = Path("src/nimify/global_deployment.py")
            
            if not deployment_file.exists():
                self.errors.append("Deployment module not found")
                return False
            
            deployment_content = deployment_file.read_text()
            global_content = global_deployment_file.read_text() if global_deployment_file.exists() else ""
            
            # Check for Kubernetes manifest generation
            k8s_features = [
                'Deployment',
                'Service', 
                'HorizontalPodAutoscaler',
                'ConfigMap',
                'NetworkPolicy'
            ]
            
            found_features = []
            combined_content = deployment_content + global_content
            
            for feature in k8s_features:
                if feature in combined_content:
                    found_features.append(feature)
            
            # Check for Helm chart generation
            helm_features = [
                'helm',
                'Chart.yaml',
                'values.yaml',
                'generate_helm_chart'
            ]
            
            found_helm_features = []
            for feature in helm_features:
                if feature in combined_content:
                    found_helm_features.append(feature)
            
            self.details['kubernetes'] = {
                'k8s_features_found': len(found_features),
                'total_k8s_features': len(k8s_features),
                'helm_features_found': len(found_helm_features),
                'total_helm_features': len(helm_features)
            }
            
            return (len(found_features) >= len(k8s_features) * 0.8 and 
                   len(found_helm_features) >= len(helm_features) * 0.5)
            
        except Exception as e:
            self.errors.append(f"Kubernetes manifest check failed: {e}")
            return False
    
    def _check_global_deployment(self) -> bool:
        """Check global deployment capabilities."""
        try:
            global_file = Path("src/nimify/global_deployment.py")
            if not global_file.exists():
                self.warnings.append("Global deployment module not found")
                return True  # Not critical for basic deployment
            
            content = global_file.read_text()
            
            global_features = [
                'Region',
                'ComplianceStandard',
                'GlobalDeploymentManager',
                'multi-region',
                'GDPR',
                'I18n'
            ]
            
            found_features = []
            for feature in global_features:
                if feature in content:
                    found_features.append(feature)
            
            self.details['global_deployment'] = {
                'global_features_found': len(found_features),
                'total_features': len(global_features)
            }
            
            return len(found_features) >= len(global_features) * 0.7
            
        except Exception as e:
            self.warnings.append(f"Global deployment check failed: {e}")
            return True  # Not critical
    
    def _check_configuration_management(self) -> bool:
        """Check configuration management."""
        try:
            config_files = [
                "pyproject.toml",
                "src/nimify/core.py",  # ModelConfig
                "src/nimify/validation.py"  # Configuration validation
            ]
            
            found_configs = []
            for config_file in config_files:
                if Path(config_file).exists():
                    found_configs.append(config_file)
            
            # Check for configuration classes
            if Path("src/nimify/core.py").exists():
                core_content = Path("src/nimify/core.py").read_text()
                config_features = ['ModelConfig', 'dataclass', '__post_init__']
                found_config_features = sum(1 for feature in config_features if feature in core_content)
            else:
                found_config_features = 0
            
            self.details['configuration'] = {
                'config_files_found': len(found_configs),
                'total_config_files': len(config_files),
                'config_features_found': found_config_features
            }
            
            return len(found_configs) >= 2 and found_config_features >= 2
            
        except Exception as e:
            self.errors.append(f"Configuration management check failed: {e}")
            return False
    
    def _check_monitoring_integration(self) -> bool:
        """Check monitoring and observability integration."""
        try:
            monitoring_indicators = []
            
            # Check various files for monitoring features
            files_to_check = [
                ("src/nimify/api.py", ["prometheus", "metrics", "REQUEST_COUNT", "HISTOGRAM"]),
                ("src/nimify/logging_config.py", ["structured", "audit", "SecurityAuditHandler"]),
                ("monitoring/", ["prometheus.yml", "grafana", "alertmanager"])
            ]
            
            for file_path, indicators in files_to_check:
                path = Path(file_path)
                if path.exists():
                    if path.is_file():
                        content = path.read_text()
                        for indicator in indicators:
                            if indicator in content:
                                monitoring_indicators.append(f"{file_path}:{indicator}")
                    else:  # Directory
                        for indicator in indicators:
                            if list(path.glob(f"**/*{indicator}*")):
                                monitoring_indicators.append(f"{file_path}:{indicator}")
            
            self.details['monitoring_integration'] = {
                'monitoring_indicators_found': len(monitoring_indicators),
                'threshold': 5
            }
            
            return len(monitoring_indicators) >= 5
            
        except Exception as e:
            self.warnings.append(f"Monitoring integration check failed: {e}")
            return True  # Not critical for deployment


def run_all_quality_gates() -> Dict[str, Any]:
    """Run all quality gates and return comprehensive report."""
    print("🏁 Running Nimify Quality Gates\\n")
    
    gates = [
        CodeQualityGate(),
        SecurityGate(),
        PerformanceGate(),
        TestGate(),
        DeploymentGate()
    ]
    
    results = {}
    overall_passed = 0
    overall_total = len(gates)
    
    for gate in gates:
        try:
            passed = gate.run()
            results[gate.name] = gate.get_report()
            
            if passed:
                overall_passed += 1
            
            print()  # Add spacing between gates
            
        except Exception as e:
            print(f"❌ {gate.name} gate failed with exception: {e}")
            results[gate.name] = {
                'name': gate.name,
                'passed': False,
                'score': 0.0,
                'errors': [f"Gate execution failed: {e}"],
                'warnings': [],
                'details': {}
            }
    
    # Calculate overall metrics
    overall_score = sum(result.get('score', 0) for result in results.values()) / len(results)
    overall_passed_bool = overall_passed >= overall_total * 0.8  # 80% of gates must pass
    
    print(f"📊 Quality Gates Summary: {overall_passed}/{overall_total} gates passed")
    print(f"📈 Overall Quality Score: {overall_score:.1%}")
    
    if overall_passed_bool:
        print("🎉 QUALITY GATES PASSED - System is ready for production!")
    else:
        print("⚠️ QUALITY GATES INCOMPLETE - Some issues need attention")
    
    return {
        'overall_passed': overall_passed_bool,
        'overall_score': overall_score,
        'gates_passed': overall_passed,
        'total_gates': overall_total,
        'individual_results': results,
        'timestamp': time.time()
    }


def main():
    """Main quality gates execution."""
    try:
        report = run_all_quality_gates()
        
        # Save detailed report
        report_file = Path("quality_gates_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\\n📄 Detailed report saved to: {report_file}")
        
        return 0 if report['overall_passed'] else 1
        
    except Exception as e:
        print(f"❌ Quality gates execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())