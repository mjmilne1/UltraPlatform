"""
Ultra Platform - Best in Class KYC & Onboarding Engine

World-class customer onboarding and verification:
- Digital-first identity verification
- Real-time document verification (OCR + AI)
- Biometric verification (liveness detection)
- AML/CTF screening (sanctions, PEPs, adverse media)
- Risk-based customer classification
- Automated decision engine (<2 min for low-risk)
- Ongoing monitoring & re-verification
- 100-point ID check compliance

Regulatory Compliance:
- AUSTRAC AML/CTF Act 2006
- ASIC RG 227 (Client Identification)
- Privacy Act 1988
- Customer Due Diligence (CDD)
- Enhanced Due Diligence (EDD)

Performance Targets:
- Onboarding Time: <2 minutes (low-risk)
- Verification SLA: <30 seconds (automated)
- False Positive Rate: <1%
- Customer Drop-off: <5%
- Straight-Through Processing: 90%+
- AML Hit Rate: 100% detection

Version: 1.0.0
"""

from .kyc_engine import (
    KYCOnboardingFramework,
    IdentityVerificationEngine,
    BiometricVerificationEngine,
    AMLScreeningEngine,
    RiskAssessmentEngine,
    OnboardingWorkflowEngine,
    CustomerProfile,
    DocumentVerification,
    BiometricVerification,
    AMLScreeningResult,
    OnboardingSession,
    DocumentType,
    VerificationStatus,
    RiskLevel,
    OnboardingStep,
    ScreeningHitType
)

__version__ = "1.0.0"

__all__ = [
    "KYCOnboardingFramework",
    "IdentityVerificationEngine",
    "BiometricVerificationEngine",
    "AMLScreeningEngine",
    "RiskAssessmentEngine",
    "OnboardingWorkflowEngine",
    "CustomerProfile",
    "DocumentVerification",
    "BiometricVerification",
    "AMLScreeningResult",
    "OnboardingSession",
    "DocumentType",
    "VerificationStatus",
    "RiskLevel",
    "OnboardingStep",
    "ScreeningHitType"
]
