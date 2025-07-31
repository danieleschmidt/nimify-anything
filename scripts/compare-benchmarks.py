#!/usr/bin/env python3
"""Compare benchmark results between two runs."""

import json
import argparse
import sys
from typing import Dict, List, Any


def load_benchmark_data(filepath: str) -> Dict[str, Any]:
    """Load benchmark data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_benchmarks(data: Dict[str, Any]) -> Dict[str, float]:
    """Extract benchmark results from data."""
    benchmarks = {}
    
    for benchmark in data.get('benchmarks', []):
        name = benchmark['name']
        # Use mean time as the primary metric
        time = benchmark['stats']['mean']
        benchmarks[name] = time
    
    return benchmarks


def compare_benchmarks(base: Dict[str, float], current: Dict[str, float], threshold: float) -> List[Dict[str, Any]]:
    """Compare benchmarks and identify regressions."""
    results = []
    
    for name in set(base.keys()) | set(current.keys()):
        if name not in base:
            results.append({
                'name': name,
                'status': 'new',
                'base_time': None,
                'current_time': current[name],
                'change_percent': None,
                'is_regression': False
            })
        elif name not in current:
            results.append({
                'name': name,
                'status': 'removed',
                'base_time': base[name],
                'current_time': None,
                'change_percent': None,
                'is_regression': False
            })
        else:
            base_time = base[name]
            current_time = current[name]
            change_percent = ((current_time - base_time) / base_time) * 100
            is_regression = change_percent > (threshold * 100)
            
            results.append({
                'name': name,
                'status': 'changed',
                'base_time': base_time,
                'current_time': current_time,
                'change_percent': change_percent,
                'is_regression': is_regression
            })
    
    return results


def format_time(time_seconds: float) -> str:
    """Format time in human-readable format."""
    if time_seconds is None:
        return "N/A"
    
    if time_seconds < 1e-6:
        return f"{time_seconds * 1e9:.2f}ns"
    elif time_seconds < 1e-3:
        return f"{time_seconds * 1e6:.2f}μs"
    elif time_seconds < 1:
        return f"{time_seconds * 1e3:.2f}ms"
    else:
        return f"{time_seconds:.2f}s"


def generate_report(results: List[Dict[str, Any]]) -> str:
    """Generate markdown report."""
    report = []
    
    # Summary
    total_tests = len(results)
    regressions = [r for r in results if r['is_regression']]
    improvements = [r for r in results if r['status'] == 'changed' and r['change_percent'] < -5]
    new_tests = [r for r in results if r['status'] == 'new']
    
    report.append("### Performance Comparison Summary")
    report.append("")
    report.append(f"- **Total benchmarks**: {total_tests}")
    report.append(f"- **Regressions**: {len(regressions)} ⚠️")
    report.append(f"- **Improvements**: {len(improvements)} ✅")
    report.append(f"- **New benchmarks**: {len(new_tests)} 🆕")
    report.append("")
    
    if regressions:
        report.append("### ⚠️ Performance Regressions")
        report.append("")
        report.append("| Benchmark | Base Time | Current Time | Change | Status |")
        report.append("|-----------|-----------|--------------|---------|--------|")
        
        for result in regressions:
            change_str = f"+{result['change_percent']:.1f}%" if result['change_percent'] > 0 else f"{result['change_percent']:.1f}%"
            report.append(f"| `{result['name']}` | {format_time(result['base_time'])} | {format_time(result['current_time'])} | {change_str} | 🔴 Slower |")
        
        report.append("")
    
    if improvements:
        report.append("### ✅ Performance Improvements")
        report.append("")
        report.append("| Benchmark | Base Time | Current Time | Change | Status |")
        report.append("|-----------|-----------|--------------|---------|--------|")
        
        for result in improvements:
            change_str = f"{result['change_percent']:.1f}%"
            report.append(f"| `{result['name']}` | {format_time(result['base_time'])} | {format_time(result['current_time'])} | {change_str} | 🟢 Faster |")
        
        report.append("")
    
    if new_tests:
        report.append("### 🆕 New Benchmarks")
        report.append("")
        report.append("| Benchmark | Time |")
        report.append("|-----------|------|")
        
        for result in new_tests:
            report.append(f"| `{result['name']}` | {format_time(result['current_time'])} |")
        
        report.append("")
    
    # Detailed results
    report.append("### Detailed Results")
    report.append("")
    report.append("| Benchmark | Base | Current | Change | Status |")
    report.append("|-----------|------|---------|---------|--------|")
    
    for result in sorted(results, key=lambda x: x['name']):
        if result['status'] == 'changed':
            change_str = f"{result['change_percent']:+.1f}%" if result['change_percent'] is not None else "N/A"
            status = "🔴" if result['is_regression'] else "🟢" if result['change_percent'] < -5 else "⚪"
        elif result['status'] == 'new':
            change_str = "NEW"
            status = "🆕"
        else:
            change_str = "REMOVED"
            status = "❌"
        
        report.append(f"| `{result['name']}` | {format_time(result['base_time'])} | {format_time(result['current_time'])} | {change_str} | {status} |")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument("base", help="Base benchmark JSON file")
    parser.add_argument("current", help="Current benchmark JSON file")
    parser.add_argument("--threshold", type=float, default=0.1, help="Regression threshold (default: 0.1 = 10%)")
    parser.add_argument("--output", help="Output file for report (optional)")
    
    args = parser.parse_args()
    
    try:
        base_data = load_benchmark_data(args.base)
        current_data = load_benchmark_data(args.current)
        
        base_benchmarks = extract_benchmarks(base_data)
        current_benchmarks = extract_benchmarks(current_data)
        
        results = compare_benchmarks(base_benchmarks, current_benchmarks, args.threshold)
        report = generate_report(results)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Report written to {args.output}")
        else:
            print(report)
        
        # Exit with error code if there are regressions
        regressions = [r for r in results if r['is_regression']]
        if regressions:
            print(f"\n⚠️  Found {len(regressions)} performance regressions!", file=sys.stderr)
            sys.exit(1)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()