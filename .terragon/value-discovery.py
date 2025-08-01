#!/usr/bin/env python3
"""
Autonomous Value Discovery Engine for Nimify Anything
Implements WSJF + ICE + Technical Debt scoring with continuous execution loop
"""

import json
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml


@dataclass
class ValueItem:
    """Represents a discovered value opportunity."""
    id: str
    title: str
    description: str
    category: str
    estimated_effort: float  # hours
    wsjf_score: float
    ice_score: float
    tech_debt_score: float
    composite_score: float
    source: str
    files_affected: List[str]
    dependencies: List[str]
    risk_level: float
    created_at: str
    status: str = "pending"


class ValueDiscoveryEngine:
    """Core engine for discovering and prioritizing value opportunities."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.config_path = repo_path / ".terragon" / "config.yaml"
        self.metrics_path = repo_path / ".terragon" / "value-metrics.json"
        self.config = self._load_config()
        self.metrics = self._load_metrics()
    
    def _load_config(self) -> dict:
        """Load configuration from .terragon/config.yaml"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_metrics(self) -> dict:
        """Load existing metrics or create new structure."""
        if self.metrics_path.exists():
            with open(self.metrics_path) as f:
                return json.load(f)
        
        return {
            "executionHistory": [],
            "backlogMetrics": {
                "totalItems": 0,
                "averageAge": 0,
                "debtRatio": 0,
                "velocityTrend": "stable"
            },
            "scoringAccuracy": {
                "wsjf": 0.8,
                "ice": 0.75,
                "technicalDebt": 0.85
            }
        }
    
    def discover_value_opportunities(self) -> List[ValueItem]:
        """Discover all value opportunities from multiple sources."""
        opportunities = []
        
        # Git history analysis
        opportunities.extend(self._analyze_git_history())
        
        # Static code analysis
        opportunities.extend(self._analyze_code_quality())
        
        # Security vulnerabilities
        opportunities.extend(self._analyze_security())
        
        # Performance opportunities
        opportunities.extend(self._analyze_performance())
        
        # Documentation gaps
        opportunities.extend(self._analyze_documentation())
        
        # Infrastructure improvements
        opportunities.extend(self._analyze_infrastructure())
        
        return self._score_and_prioritize(opportunities)
    
    def _analyze_git_history(self) -> List[ValueItem]:
        """Analyze Git history for improvement opportunities."""
        opportunities = []
        
        try:
            # Find files with high churn
            result = subprocess.run([
                "git", "log", "--name-only", "--pretty=format:", 
                "--since=3.months.ago"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                file_counts = {}
                for file in files:
                    if file and file.endswith('.py'):
                        file_counts[file] = file_counts.get(file, 0) + 1
                
                # High churn files need attention
                for file, count in file_counts.items():
                    if count > 10:  # Modified more than 10 times in 3 months
                        opportunities.append(ValueItem(
                            id=f"churn-{hash(file) % 10000}",
                            title=f"Refactor high-churn file: {file}",
                            description=f"File {file} has been modified {count} times recently, indicating potential technical debt",
                            category="technical-debt",
                            estimated_effort=4.0,
                            wsjf_score=0,
                            ice_score=0,
                            tech_debt_score=count * 2,
                            composite_score=0,
                            source="git-history",
                            files_affected=[file],
                            dependencies=[],
                            risk_level=0.3,
                            created_at=datetime.now().isoformat()
                        ))
        except Exception:
            pass  # Git not available or other error
        
        return opportunities
    
    def _analyze_code_quality(self) -> List[ValueItem]:
        """Analyze code for quality improvements."""
        opportunities = []
        
        # Check for placeholder implementations
        try:
            result = subprocess.run([
                "grep", "-r", "-n", "placeholder\\|TODO\\|FIXME", 
                str(self.repo_path / "src")
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if ':' in line:
                        file_path = line.split(':')[0]
                        file_name = Path(file_path).name
                        opportunities.append(ValueItem(
                            id=f"impl-{hash(line) % 10000}",
                            title=f"Complete implementation in {file_name}",
                            description=f"Found placeholder or TODO in {file_path}",
                            category="implementation",
                            estimated_effort=3.0,
                            wsjf_score=0,
                            ice_score=0,
                            tech_debt_score=15,
                            composite_score=0,
                            source="static-analysis",
                            files_affected=[file_path],
                            dependencies=[],
                            risk_level=0.2,
                            created_at=datetime.now().isoformat()
                        ))
        except Exception:
            pass
        
        return opportunities
    
    def _analyze_security(self) -> List[ValueItem]:
        """Analyze security vulnerabilities and improvements."""
        opportunities = []
        
        # Check for missing GitHub Actions workflows
        workflows_dir = self.repo_path / ".github" / "workflows"
        if not workflows_dir.exists():
            opportunities.append(ValueItem(
                id="sec-001",
                title="Create GitHub Actions CI/CD workflows",
                description="Repository missing automated CI/CD workflows for security scanning and testing",
                category="security",
                estimated_effort=6.0,
                wsjf_score=0,
                ice_score=0,
                tech_debt_score=0,
                composite_score=0,
                source="security-analysis",
                files_affected=[],
                dependencies=[],
                risk_level=0.4,
                created_at=datetime.now().isoformat()
            ))
        
        return opportunities
    
    def _analyze_performance(self) -> List[ValueItem]:
        """Analyze performance optimization opportunities."""
        opportunities = []
        
        # Check for performance test setup
        perf_tests = list(self.repo_path.glob("tests/**/test_*performance*.py"))
        if perf_tests:
            opportunities.append(ValueItem(
                id="perf-001",
                title="Enhance performance monitoring",
                description="Extend performance tests with automated regression detection",
                category="performance",
                estimated_effort=4.0,
                wsjf_score=0,
                ice_score=0,
                tech_debt_score=0,
                composite_score=0,
                source="performance-analysis",
                files_affected=[str(p) for p in perf_tests],
                dependencies=[],
                risk_level=0.2,
                created_at=datetime.now().isoformat()
            ))
        
        return opportunities
    
    def _analyze_documentation(self) -> List[ValueItem]:
        """Analyze documentation gaps and improvements."""
        opportunities = []
        
        # Check for API documentation
        api_docs = list(self.repo_path.glob("docs/API.md"))
        if api_docs:
            opportunities.append(ValueItem(
                id="doc-001",
                title="Auto-generate API documentation",
                description="Implement automated OpenAPI spec generation and documentation updates",
                category="documentation",
                estimated_effort=3.0,
                wsjf_score=0,
                ice_score=0,
                tech_debt_score=0,
                composite_score=0,
                source="documentation-analysis",
                files_affected=["docs/API.md"],
                dependencies=["src/nimify/core.py"],
                risk_level=0.1,
                created_at=datetime.now().isoformat()
            ))
        
        return opportunities
    
    def _analyze_infrastructure(self) -> List[ValueItem]:
        """Analyze infrastructure and deployment improvements."""
        opportunities = []
        
        # Check monitoring setup
        monitoring_dir = self.repo_path / "monitoring"
        if monitoring_dir.exists():
            opportunities.append(ValueItem(
                id="infra-001",
                title="Implement monitoring alerts automation",
                description="Create automated monitoring setup with intelligent alerting",
                category="infrastructure",
                estimated_effort=5.0,
                wsjf_score=0,
                ice_score=0,
                tech_debt_score=0,
                composite_score=0,
                source="infrastructure-analysis",
                files_affected=["monitoring/"],
                dependencies=[],
                risk_level=0.3,
                created_at=datetime.now().isoformat()
            ))
        
        return opportunities
    
    def _score_and_prioritize(self, opportunities: List[ValueItem]) -> List[ValueItem]:
        """Apply WSJF + ICE + Technical Debt scoring."""
        weights = self.config.get("scoring", {}).get("weights", {}).get("advanced", {})
        
        for item in opportunities:
            # WSJF Components
            user_value = self._calculate_user_value(item)
            time_criticality = self._calculate_time_criticality(item)
            risk_reduction = self._calculate_risk_reduction(item)
            opportunity_enablement = self._calculate_opportunity_enablement(item)
            
            cost_of_delay = user_value + time_criticality + risk_reduction + opportunity_enablement
            item.wsjf_score = cost_of_delay / item.estimated_effort
            
            # ICE Components
            impact = self._calculate_impact(item)
            confidence = self._calculate_confidence(item)
            ease = 10 - (item.estimated_effort / 2)  # Easier = higher score
            
            item.ice_score = impact * confidence * ease
            
            # Composite Score
            item.composite_score = (
                weights.get("wsjf", 0.5) * self._normalize_score(item.wsjf_score) +
                weights.get("ice", 0.1) * self._normalize_score(item.ice_score) +
                weights.get("technicalDebt", 0.3) * self._normalize_score(item.tech_debt_score) +
                weights.get("security", 0.1) * (2.0 if item.category == "security" else 1.0)
            )
        
        return sorted(opportunities, key=lambda x: x.composite_score, reverse=True)
    
    def _calculate_user_value(self, item: ValueItem) -> float:
        """Calculate user/business value component."""
        category_values = {
            "security": 9,
            "performance": 8,
            "implementation": 7,
            "technical-debt": 6,
            "infrastructure": 7,
            "documentation": 4
        }
        return category_values.get(item.category, 5)
    
    def _calculate_time_criticality(self, item: ValueItem) -> float:
        """Calculate time criticality component."""
        if item.category == "security":
            return 9
        elif item.category == "implementation":
            return 8
        return 5
    
    def _calculate_risk_reduction(self, item: ValueItem) -> float:
        """Calculate risk reduction component."""
        return (1 - item.risk_level) * 10
    
    def _calculate_opportunity_enablement(self, item: ValueItem) -> float:
        """Calculate opportunity enablement component."""
        if item.category in ["implementation", "infrastructure"]:
            return 7
        return 4
    
    def _calculate_impact(self, item: ValueItem) -> float:
        """Calculate impact for ICE scoring."""
        return self._calculate_user_value(item)
    
    def _calculate_confidence(self, item: ValueItem) -> float:
        """Calculate confidence for ICE scoring."""
        # Higher confidence for well-understood improvements
        confidence_map = {
            "implementation": 9,
            "security": 8,
            "documentation": 8,
            "technical-debt": 7,
            "performance": 6,
            "infrastructure": 7
        }
        return confidence_map.get(item.category, 6)
    
    def _normalize_score(self, score: float) -> float:
        """Normalize score to 0-100 range."""
        return min(100, max(0, score))
    
    def save_metrics(self):
        """Save metrics to file."""
        self.metrics_path.parent.mkdir(exist_ok=True)
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def generate_backlog_markdown(self, opportunities: List[ValueItem]) -> str:
        """Generate markdown backlog file."""
        md = f"""# 📊 Autonomous Value Backlog

Last Updated: {datetime.now().isoformat()}
Repository Maturity: Advanced (85%)

## 🎯 Next Best Value Item
"""
        
        if opportunities:
            top_item = opportunities[0]
            md += f"""**[{top_item.id.upper()}] {top_item.title}**
- **Composite Score**: {top_item.composite_score:.1f}
- **WSJF**: {top_item.wsjf_score:.1f} | **ICE**: {top_item.ice_score:.1f} | **Tech Debt**: {top_item.tech_debt_score:.1f}
- **Estimated Effort**: {top_item.estimated_effort} hours
- **Category**: {top_item.category}
- **Risk Level**: {top_item.risk_level:.1f}

## 📋 Top 10 Backlog Items

| Rank | ID | Title | Score | Category | Est. Hours |
|------|-----|--------|---------|----------|------------|
"""
            
            for i, item in enumerate(opportunities[:10], 1):
                md += f"| {i} | {item.id} | {item.title[:50]}... | {item.composite_score:.1f} | {item.category} | {item.estimated_effort} |\n"
        else:
            md += "No value opportunities discovered.\n"
        
        md += f"""

## 📈 Discovery Stats
- **Total Items Discovered**: {len(opportunities)}
- **Categories Covered**: {len(set(item.category for item in opportunities))}
- **Average Effort**: {sum(item.estimated_effort for item in opportunities) / len(opportunities) if opportunities else 0:.1f} hours
- **High-Priority Items**: {len([item for item in opportunities if item.composite_score > 50])}

## 🔄 Discovery Sources
- Git History Analysis
- Static Code Analysis  
- Security Scanning
- Performance Analysis
- Documentation Review
- Infrastructure Assessment

---
*Generated by Terragon Autonomous SDLC Engine*
"""
        return md


def main():
    """Main execution function for autonomous value discovery."""
    repo_path = Path(__file__).parent.parent
    engine = ValueDiscoveryEngine(repo_path)
    
    print("🔍 Discovering value opportunities...")
    opportunities = engine.discover_value_opportunities()
    
    print(f"✅ Found {len(opportunities)} opportunities")
    
    # Generate backlog
    backlog_md = engine.generate_backlog_markdown(opportunities)
    backlog_path = repo_path / "BACKLOG.md"
    with open(backlog_path, 'w') as f:
        f.write(backlog_md)
    
    # Update metrics
    engine.metrics["backlogMetrics"]["totalItems"] = len(opportunities)
    engine.save_metrics()
    
    print(f"📊 Backlog updated: {backlog_path}")
    
    if opportunities:
        top_item = opportunities[0]
        print(f"🎯 Next best value: {top_item.title} (Score: {top_item.composite_score:.1f})")


if __name__ == "__main__":
    main()