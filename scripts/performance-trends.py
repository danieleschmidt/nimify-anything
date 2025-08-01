#!/usr/bin/env python3
"""
Performance Trends Analysis for Autonomous SDLC
Generates comprehensive performance trend reports with visualizations
"""

import json
import argparse
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Tuple


class PerformanceTrendAnalyzer:
    """Analyze performance trends and generate comprehensive reports."""
    
    def __init__(self):
        self.history_path = Path(".terragon/performance-history.json")
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load performance history."""
        if self.history_path.exists():
            with open(self.history_path) as f:
                return json.load(f)
        return []
    
    def analyze_trends(self, lookback_days: int = 30) -> Dict[str, Any]:
        """Analyze performance trends over specified period."""
        if not self.history:
            return {"status": "no_data", "message": "No performance history available"}
        
        # Filter entries by lookback period
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_entries = [
            entry for entry in self.history
            if datetime.fromisoformat(entry['timestamp']) >= cutoff_date
        ]
        
        if not recent_entries:
            return {"status": "no_recent_data", "message": f"No data in last {lookback_days} days"}
        
        analysis = {
            "status": "analyzed",
            "period_days": lookback_days,
            "total_entries": len(recent_entries),
            "date_range": {
                "start": recent_entries[0]['timestamp'],
                "end": recent_entries[-1]['timestamp']
            },
            "overall_trends": self._analyze_overall_trends(recent_entries),
            "benchmark_trends": self._analyze_benchmark_trends(recent_entries),
            "performance_insights": self._generate_insights(recent_entries),
            "recommendations": self._generate_recommendations(recent_entries)
        }
        
        return analysis
    
    def _analyze_overall_trends(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall performance trends."""
        avg_times = [entry['average_time'] for entry in entries]
        total_times = [entry['total_time'] for entry in entries]
        benchmark_counts = [entry['total_benchmarks'] for entry in entries]
        
        return {
            "average_execution_time": {
                "current": avg_times[-1] if avg_times else 0,
                "mean": statistics.mean(avg_times) if avg_times else 0,
                "median": statistics.median(avg_times) if avg_times else 0,
                "trend_percent": self._calculate_trend_percentage(avg_times),
                "stability": self._calculate_stability(avg_times)
            },
            "total_execution_time": {
                "current": total_times[-1] if total_times else 0,
                "mean": statistics.mean(total_times) if total_times else 0,
                "trend_percent": self._calculate_trend_percentage(total_times)
            },
            "benchmark_coverage": {
                "current": benchmark_counts[-1] if benchmark_counts else 0,
                "mean": statistics.mean(benchmark_counts) if benchmark_counts else 0,
                "trend": "stable" if len(set(benchmark_counts)) <= 2 else "variable"
            }
        }
    
    def _analyze_benchmark_trends(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze individual benchmark trends."""
        # Collect all unique benchmarks
        all_benchmarks = set()
        for entry in entries:
            all_benchmarks.update(entry['benchmarks'].keys())
        
        benchmark_analysis = {}
        
        for benchmark_name in all_benchmarks:
            values = []
            coverage = 0
            
            for entry in entries:
                if benchmark_name in entry['benchmarks']:
                    values.append(entry['benchmarks'][benchmark_name])
                    coverage += 1
            
            if values:
                benchmark_analysis[benchmark_name] = {
                    "coverage_percent": (coverage / len(entries)) * 100,
                    "current_time": values[-1],
                    "average_time": statistics.mean(values),
                    "trend_percent": self._calculate_trend_percentage(values),
                    "stability": self._calculate_stability(values),
                    "performance_category": self._categorize_performance(
                        values[-1], statistics.mean(values)
                    )
                }
        
        return benchmark_analysis
    
    def _calculate_trend_percentage(self, values: List[float]) -> float:
        """Calculate trend percentage (positive = getting worse)."""
        if len(values) < 2:
            return 0.0
        
        # Use linear regression slope for trend
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(values)
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        
        # Convert slope to percentage change
        if y_mean != 0:
            return (slope * (n - 1) / y_mean) * 100
        return 0.0
    
    def _calculate_stability(self, values: List[float]) -> str:
        """Calculate performance stability rating."""
        if len(values) < 3:
            return "insufficient_data"
        
        mean_val = statistics.mean(values)
        if mean_val == 0:
            return "stable"
        
        cv = statistics.stdev(values) / mean_val  # Coefficient of variation
        
        if cv < 0.05:
            return "very_stable"
        elif cv < 0.15:
            return "stable"
        elif cv < 0.30:
            return "moderate"
        else:
            return "unstable"
    
    def _categorize_performance(self, current: float, average: float) -> str:
        """Categorize current performance relative to average."""
        if average == 0:
            return "unknown"
        
        ratio = current / average
        
        if ratio < 0.90:
            return "excellent"
        elif ratio < 1.10:
            return "good"
        elif ratio < 1.25:
            return "fair"
        else:
            return "poor"
    
    def _generate_insights(self, entries: List[Dict[str, Any]]) -> List[str]:
        """Generate performance insights."""
        insights = []
        
        if len(entries) < 3:
            insights.append("Insufficient data for meaningful trend analysis")
            return insights
        
        # Overall performance insight
        avg_times = [entry['average_time'] for entry in entries]
        overall_trend = self._calculate_trend_percentage(avg_times)
        
        if abs(overall_trend) < 5:
            insights.append("Overall performance is stable with minimal variance")
        elif overall_trend > 15:
            insights.append(f"Performance has degraded by {overall_trend:.1f}% - investigation recommended")
        elif overall_trend < -15:
            insights.append(f"Performance has improved by {abs(overall_trend):.1f}% - excellent progress")
        
        # Stability insight
        stability = self._calculate_stability(avg_times)
        if stability == "unstable":
            insights.append("Performance shows high variability - consider investigating environmental factors")
        elif stability == "very_stable":
            insights.append("Performance is very consistent - well-optimized system")
        
        # Coverage insight
        benchmark_counts = [entry['total_benchmarks'] for entry in entries]
        if len(set(benchmark_counts)) > 3:
            insights.append("Benchmark coverage is inconsistent - ensure stable test suite")
        
        return insights
    
    def _generate_recommendations(self, entries: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if len(entries) < 5:
            recommendations.append("Collect more performance data for better trend analysis")
            return recommendations
        
        avg_times = [entry['average_time'] for entry in entries]
        trend = self._calculate_trend_percentage(avg_times)
        stability = self._calculate_stability(avg_times)
        
        # Trend-based recommendations
        if trend > 20:
            recommendations.append("🔴 URGENT: Investigate performance regression - >20% degradation")
            recommendations.append("- Profile recent code changes")
            recommendations.append("- Check for memory leaks")
            recommendations.append("- Review dependency updates")
        elif trend > 10:
            recommendations.append("🟡 Monitor performance closely - moderate degradation detected")
            recommendations.append("- Run detailed profiling")
            recommendations.append("- Consider performance optimizations")
        elif trend < -10:
            recommendations.append("🟢 Document performance improvements for knowledge sharing")
        
        # Stability-based recommendations
        if stability == "unstable":
            recommendations.append("🔧 Improve test environment consistency")
            recommendations.append("- Use dedicated performance testing infrastructure")
            recommendations.append("- Isolate performance tests from other processes")
        
        # General recommendations
        if len(entries) > 20:
            recommendations.append("📊 Consider implementing automated performance gates in CI/CD")
            recommendations.append("📈 Set up performance monitoring dashboards")
        
        return recommendations
    
    def generate_html_report(self, analysis: Dict[str, Any], output_path: str):
        """Generate HTML performance report."""
        html_template = '''
<!DOCTYPE html>
<html>
<head>
    <title>Performance Trends Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { background: #f4f4f4; padding: 20px; border-radius: 5px; }
        .metric { background: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .trend-good { color: #27ae60; }
        .trend-bad { color: #e74c3c; }
        .trend-neutral { color: #7f8c8d; }
        .benchmark { margin: 10px 0; padding: 10px; border-left: 3px solid #3498db; }
        .insight { background: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; }
        .recommendation { background: #d4edda; padding: 10px; margin: 5px 0; border-radius: 3px; }
        .recommendation.urgent { background: #f8d7da; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Performance Trends Report</h1>
        <p><strong>Analysis Period:</strong> {period_days} days</p>
        <p><strong>Total Entries:</strong> {total_entries}</p>
        <p><strong>Date Range:</strong> {date_start} to {date_end}</p>
        <p><strong>Generated:</strong> {generated_time}</p>
    </div>
    
    <h2>🎯 Overall Performance Trends</h2>
    <div class="metric">
        <h3>Average Execution Time</h3>
        <p><strong>Current:</strong> {avg_current:.4f}s</p>
        <p><strong>Mean:</strong> {avg_mean:.4f}s</p>
        <p><strong>Trend:</strong> <span class="{avg_trend_class}">{avg_trend:+.1f}%</span></p>
        <p><strong>Stability:</strong> {avg_stability}</p>
    </div>
    
    <h2>📈 Individual Benchmark Trends</h2>
    <table>
        <thead>
            <tr>
                <th>Benchmark</th>
                <th>Current Time (s)</th>
                <th>Average Time (s)</th>
                <th>Trend (%)</th>
                <th>Stability</th>
                <th>Category</th>
            </tr>
        </thead>
        <tbody>
            {benchmark_rows}
        </tbody>
    </table>
    
    <h2>💡 Performance Insights</h2>
    {insights_html}
    
    <h2>🎯 Recommendations</h2>
    {recommendations_html}
    
    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #666;">
        <p>Generated by Terragon Autonomous SDLC Performance Monitor</p>
    </footer>
</body>
</html>
        '''
        
        # Format template
        overall = analysis['overall_trends']['average_execution_time']
        trend_class = 'trend-bad' if overall['trend_percent'] > 5 else 'trend-good' if overall['trend_percent'] < -5 else 'trend-neutral'
        
        # Generate benchmark rows
        benchmark_rows = ''
        for name, data in analysis['benchmark_trends'].items():
            trend_class_bench = 'trend-bad' if data['trend_percent'] > 5 else 'trend-good' if data['trend_percent'] < -5 else 'trend-neutral'
            benchmark_rows += f'''
            <tr>
                <td>{name}</td>
                <td>{data['current_time']:.4f}</td>
                <td>{data['average_time']:.4f}</td>
                <td><span class="{trend_class_bench}">{data['trend_percent']:+.1f}</span></td>
                <td>{data['stability']}</td>
                <td>{data['performance_category']}</td>
            </tr>
            '''
        
        # Generate insights HTML
        insights_html = ''.join(f'<div class="insight">{insight}</div>' for insight in analysis['performance_insights'])
        
        # Generate recommendations HTML
        recommendations_html = ''
        for rec in analysis['recommendations']:
            rec_class = 'recommendation urgent' if '🔴 URGENT' in rec else 'recommendation'
            recommendations_html += f'<div class="{rec_class}">{rec}</div>'
        
        html_content = html_template.format(
            period_days=analysis['period_days'],
            total_entries=analysis['total_entries'],
            date_start=analysis['date_range']['start'][:10],
            date_end=analysis['date_range']['end'][:10] if analysis['date_range']['end'] else 'Present',
            generated_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            avg_current=overall['current'],
            avg_mean=overall['mean'],
            avg_trend=overall['trend_percent'],
            avg_trend_class=trend_class,
            avg_stability=overall['stability'],
            benchmark_rows=benchmark_rows,
            insights_html=insights_html,
            recommendations_html=recommendations_html
        )
        
        with open(output_path, 'w') as f:
            f.write(html_content)


def main():
    """Main function for performance trend analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze performance trends and generate reports"
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to performance results directory'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output path for HTML report'
    )
    parser.add_argument(
        '--lookback-days',
        type=int,
        default=30,
        help='Days to analyze (default: 30)'
    )
    
    args = parser.parse_args()
    
    analyzer = PerformanceTrendAnalyzer()
    
    print(f"📊 Analyzing performance trends over {args.lookback_days} days...")
    analysis = analyzer.analyze_trends(args.lookback_days)
    
    if analysis['status'] != 'analyzed':
        print(f"❌ {analysis['message']}")
        return 1
    
    print(f"✅ Analyzed {analysis['total_entries']} performance entries")
    
    # Generate insights summary
    overall_trend = analysis['overall_trends']['average_execution_time']['trend_percent']
    if abs(overall_trend) > 10:
        trend_direction = "degraded" if overall_trend > 0 else "improved"
        print(f"📈 Performance has {trend_direction} by {abs(overall_trend):.1f}%")
    else:
        print("📊 Performance is stable")
    
    # Generate HTML report
    print(f"📝 Generating HTML report: {args.output}")
    analyzer.generate_html_report(analysis, args.output)
    print("✅ Report generated successfully")
    
    return 0


if __name__ == "__main__":
    exit(main())