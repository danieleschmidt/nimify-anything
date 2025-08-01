#!/usr/bin/env python3
"""
Performance Health Check for Autonomous SDLC
Monitors performance trends and triggers alerts for degradation
"""

import json
import argparse
import smtplib
import statistics
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from typing import Dict, List, Any, Optional
from pathlib import Path


class PerformanceHealthMonitor:
    """Monitor performance health and detect degradation patterns."""
    
    def __init__(self, alert_threshold: float = 20.0):
        self.alert_threshold = alert_threshold  # % performance degradation
        self.history_path = Path(".terragon/performance-history.json")
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load performance history."""
        if self.history_path.exists():
            with open(self.history_path) as f:
                return json.load(f)
        return []
    
    def _save_history(self):
        """Save performance history."""
        self.history_path.parent.mkdir(exist_ok=True)
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def add_benchmark_results(self, results_file: str):
        """Add new benchmark results to history."""
        with open(results_file) as f:
            data = json.load(f)
        
        # Extract key metrics
        total_time = sum(b['stats']['mean'] for b in data['benchmarks'])
        avg_time = total_time / len(data['benchmarks'])
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'total_benchmarks': len(data['benchmarks']),
            'total_time': total_time,
            'average_time': avg_time,
            'benchmarks': {
                b['name']: b['stats']['mean'] 
                for b in data['benchmarks']
            }
        }
        
        self.history.append(entry)
        
        # Keep only last 100 entries
        if len(self.history) > 100:
            self.history = self.history[-100:]
            
        self._save_history()
        return entry
    
    def check_performance_trends(self, lookback_days: int = 7) -> Dict[str, Any]:
        """Check for performance degradation trends."""
        if len(self.history) < 2:
            return {"status": "insufficient_data", "alerts": []}
        
        # Filter recent entries
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_entries = [
            entry for entry in self.history
            if datetime.fromisoformat(entry['timestamp']) >= cutoff_date
        ]
        
        if len(recent_entries) < 2:
            return {"status": "insufficient_recent_data", "alerts": []}
        
        alerts = []
        
        # Check overall performance trend
        recent_avg_times = [entry['average_time'] for entry in recent_entries]
        if len(recent_avg_times) >= 3:
            trend = self._calculate_trend(recent_avg_times)
            if trend > self.alert_threshold:
                alerts.append({
                    "type": "overall_degradation",
                    "severity": "high" if trend > 50 else "medium",
                    "message": f"Overall performance degraded by {trend:.1f}% over {lookback_days} days",
                    "trend_percent": trend
                })
        
        # Check individual benchmark trends
        benchmark_alerts = self._check_individual_benchmarks(recent_entries)
        alerts.extend(benchmark_alerts)
        
        # Check for memory issues
        memory_alerts = self._check_memory_patterns(recent_entries)
        alerts.extend(memory_alerts)
        
        return {
            "status": "analyzed",
            "alerts": alerts,
            "recent_entries": len(recent_entries),
            "trend_analysis": {
                "average_time_trend": trend if 'trend' in locals() else 0,
                "performance_direction": "degrading" if alerts else "stable"
            }
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend percentage (positive = getting worse)."""
        if len(values) < 2:
            return 0
        
        # Compare recent average to older average
        mid_point = len(values) // 2
        older_avg = statistics.mean(values[:mid_point])
        recent_avg = statistics.mean(values[mid_point:])
        
        if older_avg == 0:
            return 0
        
        return ((recent_avg - older_avg) / older_avg) * 100
    
    def _check_individual_benchmarks(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check individual benchmark performance."""
        alerts = []
        
        if len(entries) < 3:
            return alerts
        
        # Get common benchmarks across all entries
        all_benchmarks = set()
        for entry in entries:
            all_benchmarks.update(entry['benchmarks'].keys())
        
        for benchmark_name in all_benchmarks:
            # Collect values for this benchmark
            values = []
            for entry in entries:
                if benchmark_name in entry['benchmarks']:
                    values.append(entry['benchmarks'][benchmark_name])
            
            if len(values) >= 3:
                trend = self._calculate_trend(values)
                if trend > self.alert_threshold:
                    alerts.append({
                        "type": "benchmark_degradation",
                        "severity": "high" if trend > 50 else "medium",
                        "message": f"Benchmark '{benchmark_name}' degraded by {trend:.1f}%",
                        "benchmark": benchmark_name,
                        "trend_percent": trend
                    })
        
        return alerts
    
    def _check_memory_patterns(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for memory-related performance issues."""
        alerts = []
        
        # Look for sudden spikes in execution time (potential memory issues)
        avg_times = [entry['average_time'] for entry in entries]
        if len(avg_times) >= 5:
            recent_avg = statistics.mean(avg_times[-3:])
            baseline_avg = statistics.mean(avg_times[:-3])
            
            if baseline_avg > 0 and recent_avg > baseline_avg * 2:
                alerts.append({
                    "type": "potential_memory_issue",
                    "severity": "high",
                    "message": "Sudden performance spike detected - possible memory leak",
                    "spike_factor": recent_avg / baseline_avg
                })
        
        return alerts
    
    def send_alert_email(self, alerts: List[Dict[str, Any]], email_endpoint: str):
        """Send performance alert email."""
        if not alerts or not email_endpoint:
            return
        
        subject = f"🚨 Performance Alert - {len(alerts)} issues detected"
        
        body = "Performance Health Check Alert\n"
        body += "=" * 40 + "\n\n"
        
        for alert in alerts:
            body += f"⚠️  {alert['severity'].upper()}: {alert['message']}\n"
            if 'trend_percent' in alert:
                body += f"   Trend: {alert['trend_percent']:.1f}% degradation\n"
            body += "\n"
        
        body += f"\nTimestamp: {datetime.now().isoformat()}\n"
        body += f"Repository: {Path.cwd().name}\n"
        
        # Create email
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = 'performance-monitor@terragon.dev'
        msg['To'] = email_endpoint
        
        try:
            # In a real implementation, you'd configure SMTP
            print(f"📧 Alert email would be sent to {email_endpoint}")
            print(f"Subject: {subject}")
            print(f"Body preview: {body[:200]}...")
        except Exception as e:
            print(f"Failed to send alert email: {e}")


def main():
    """Main function for performance health checking."""
    parser = argparse.ArgumentParser(
        description="Monitor performance health and detect degradation"
    )
    parser.add_argument(
        '--results', 
        required=True,
        help='Path to benchmark results JSON file'
    )
    parser.add_argument(
        '--alert-threshold',
        type=float,
        default=20.0,
        help='Performance degradation threshold in percent (default: 20)'
    )
    parser.add_argument(
        '--email-endpoint',
        help='Email address for alerts'
    )
    parser.add_argument(
        '--lookback-days',
        type=int,
        default=7,
        help='Days to look back for trend analysis (default: 7)'
    )
    
    args = parser.parse_args()
    
    monitor = PerformanceHealthMonitor(args.alert_threshold)
    
    # Add current results to history
    print(f"📊 Adding benchmark results from {args.results}")
    entry = monitor.add_benchmark_results(args.results)
    print(f"✅ Added {entry['total_benchmarks']} benchmarks with avg time {entry['average_time']:.4f}s")
    
    # Check for performance trends
    print(f"🔍 Analyzing performance trends over {args.lookback_days} days...")
    analysis = monitor.check_performance_trends(args.lookback_days)
    
    if analysis['status'] == 'insufficient_data':
        print("ℹ️  Insufficient data for trend analysis")
        return
    
    if analysis['alerts']:
        print(f"🚨 Found {len(analysis['alerts'])} performance alerts:")
        for alert in analysis['alerts']:
            severity_emoji = "🔥" if alert['severity'] == 'high' else "⚠️"
            print(f"  {severity_emoji} {alert['message']}")
        
        # Send email alerts if configured
        if args.email_endpoint:
            monitor.send_alert_email(analysis['alerts'], args.email_endpoint)
        
        # Exit with error code if high severity alerts
        high_severity_alerts = [a for a in analysis['alerts'] if a['severity'] == 'high']
        if high_severity_alerts:
            print(f"💥 {len(high_severity_alerts)} high-severity performance issues detected!")
            return 1
    else:
        print("✅ No performance degradation detected")
        trend_direction = analysis['trend_analysis']['performance_direction']
        print(f"📈 Performance trend: {trend_direction}")
    
    return 0


if __name__ == "__main__":
    exit(main())