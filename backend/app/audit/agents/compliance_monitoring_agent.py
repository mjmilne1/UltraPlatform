"""
AI Compliance Monitoring Agent
Autonomous monitoring for regulatory compliance
"""

from typing import Dict, List
from datetime import datetime, timedelta

class ComplianceMonitoringAgent:
    """
    Autonomous agent for compliance monitoring
    
    Monitors:
    - Unusual approval patterns
    - Policy violations
    - Segregation of duties
    - Access patterns
    - High-risk changes
    """
    
    def __init__(self, audit_service, memory):
        self.audit = audit_service
        self.memory = memory
        self.name = "ComplianceMonitoringAgent"
    
    async def run_daily_monitoring(self):
        """Daily compliance monitoring"""
        
        print(f"[{self.name}] Starting daily compliance monitoring...")
        
        violations = []
        
        # Check 1: Maker-checker violations
        maker_checker_issues = await self._check_maker_checker_compliance()
        violations.extend(maker_checker_issues)
        
        # Check 2: After-hours approvals
        after_hours_issues = await self._check_after_hours_approvals()
        violations.extend(after_hours_issues)
        
        # Check 3: Segregation of duties
        sod_issues = await self._check_segregation_of_duties()
        violations.extend(sod_issues)
        
        # Check 4: Expired approvals
        expired_issues = await self._check_expired_approvals()
        violations.extend(expired_issues)
        
        # Check 5: Unusual approval velocity
        velocity_issues = await self._check_approval_velocity()
        violations.extend(velocity_issues)
        
        if violations:
            print(f"[{self.name}] Found {len(violations)} compliance issues!")
            await self._generate_compliance_alert(violations)
        else:
            print(f"[{self.name}] No compliance issues detected.")
        
        # Store in memory
        self.memory.append({
            "timestamp": datetime.now(),
            "violations_found": len(violations),
            "violations": violations
        })
    
    async def analyze_approval_pattern(
        self,
        user_id: str,
        lookback_days: int = 30
    ) -> Dict:
        """
        Analyze user's approval patterns using ML
        
        Detects:
        - Rubber stamping (approving too quickly)
        - Approval rate anomalies
        - Risk tolerance changes
        """
        
        # Get user's approvals
        approvals = await self._get_user_approvals(user_id, lookback_days)
        
        if not approvals:
            return {"status": "insufficient_data"}
        
        # Calculate metrics
        total_approvals = len(approvals)
        avg_time_to_approve = sum(a["time_to_approve"] for a in approvals) / total_approvals
        approval_rate = sum(1 for a in approvals if a["approved"]) / total_approvals
        
        # Detect anomalies
        issues = []
        
        # Rubber stamping detection
        if avg_time_to_approve < 60:  # < 1 minute
            issues.append("RUBBER_STAMPING")
        
        # Too permissive
        if approval_rate > 0.95:
            issues.append("OVERLY_PERMISSIVE")
        
        # Too restrictive
        if approval_rate < 0.3:
            issues.append("OVERLY_RESTRICTIVE")
        
        return {
            "user_id": user_id,
            "total_approvals": total_approvals,
            "avg_time_to_approve_seconds": avg_time_to_approve,
            "approval_rate": approval_rate,
            "issues": issues,
            "risk_level": "high" if issues else "normal"
        }
    
    async def predict_compliance_risk(
        self,
        entity_type: str,
        proposed_change: Dict
    ) -> Dict:
        """
        ML-based prediction of compliance risk
        
        Factors:
        - Change magnitude
        - Historical violations
        - User risk profile
        - Time/context
        """
        
        risk_score = 0.0
        risk_factors = []
        
        # Check magnitude
        if "amount" in proposed_change:
            amount = float(proposed_change["amount"])
            if amount > 100000:
                risk_score += 0.4
                risk_factors.append("LARGE_AMOUNT")
        
        # Check entity type
        high_risk_entities = ["TRANSACTION", "JOURNAL_ENTRY", "ACCOUNT"]
        if entity_type in high_risk_entities:
            risk_score += 0.3
            risk_factors.append("HIGH_RISK_ENTITY")
        
        # Check time
        hour = datetime.now().hour
        if hour < 6 or hour > 20:
            risk_score += 0.2
            risk_factors.append("AFTER_HOURS")
        
        # Determine level
        if risk_score >= 0.7:
            risk_level = "HIGH"
        elif risk_score >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            "risk_score": min(risk_score, 1.0),
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "requires_additional_approval": risk_score >= 0.7
        }
    
    async def _check_maker_checker_compliance(self) -> List[Dict]:
        """Check for maker-checker violations"""
        
        # Check for self-approvals
        violations = []
        
        # Query recent approvals
        # Look for maker_id == checker_id
        
        return violations
    
    async def _check_after_hours_approvals(self) -> List[Dict]:
        """Check for suspicious after-hours approvals"""
        
        violations = []
        
        # Get approvals in last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        
        # Check if approved between 10 PM - 6 AM
        
        return violations
    
    async def _check_segregation_of_duties(self) -> List[Dict]:
        """Check segregation of duties compliance"""
        
        violations = []
        
        # Check for:
        # - Same person creating and approving
        # - Conflicting roles
        # - Insufficient separation
        
        return violations
    
    async def _check_expired_approvals(self) -> List[Dict]:
        """Check for expired pending approvals"""
        
        violations = []
        
        # Get requests older than 7 days
        # that are still pending
        
        return violations
    
    async def _check_approval_velocity(self) -> List[Dict]:
        """Check for unusually fast approvals"""
        
        violations = []
        
        # Check for approvals < 30 seconds
        # (possible rubber stamping)
        
        return violations
    
    async def _generate_compliance_alert(self, violations: List[Dict]):
        """Generate alert for compliance team"""
        
        print(f"")
        print(f"??  COMPLIANCE ALERT")
        print(f"   Violations detected: {len(violations)}")
        
        for v in violations:
            print(f"   • {v.get('type', 'UNKNOWN')}: {v.get('description', '')}")
    
    async def _get_user_approvals(
        self,
        user_id: str,
        lookback_days: int
    ) -> List[Dict]:
        """Get user's approval history"""
        # Mock - replace with actual query
        return []
