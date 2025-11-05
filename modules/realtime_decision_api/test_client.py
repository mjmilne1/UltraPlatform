import requests
import json
import time
from typing import Dict, List
from datetime import datetime

class DecisionAPIClient:
    '''Client for testing Real-Time Decision API'''
    
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "test_api_key_12345"):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def health_check(self):
        '''Check API health'''
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def make_decision(self, decision_request: Dict):
        '''Make a single decision'''
        response = self.session.post(
            f"{self.base_url}/decision",
            json=decision_request
        )
        return response.json()
    
    def batch_decision(self, decision_requests: List[Dict]):
        '''Make batch decisions'''
        response = self.session.post(
            f"{self.base_url}/batch-decision",
            json=decision_requests
        )
        return response.json()
    
    def simulate(self, **kwargs):
        '''Simulate a decision'''
        response = self.session.post(
            f"{self.base_url}/simulate",
            params=kwargs
        )
        return response.json()
    
    def get_metrics(self):
        '''Get API metrics'''
        response = self.session.get(f"{self.base_url}/metrics")
        return response.json()

def run_tests():
    '''Run API tests'''
    print("🧪 TESTING REAL-TIME DECISION API")
    print("="*80)
    
    client = DecisionAPIClient()
    
    # Test 1: Health Check
    print("\n1️⃣ Health Check Test")
    print("-"*40)
    try:
        health = client.health_check()
        print(f"✅ API Status: {health['status']}")
        print(f"   Version: {health['version']}")
        print(f"   Uptime: {health['uptime_seconds']:.0f} seconds")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test 2: Credit Approval Decision
    print("\n2️⃣ Credit Approval Test")
    print("-"*40)
    
    credit_request = {
        "decision_type": "credit_approval",
        "customer_data": {
            "customer_id": "TEST_001",
            "annual_income": 85000,
            "credit_score": 720,
            "existing_debt": 25000,
            "monthly_debt_payments": 1500,
            "requested_amount": 30000,
            "requested_term_months": 60,
            "employment_status": "employed",
            "employment_duration_months": 48
        }
    }
    
    try:
        response = client.make_decision(credit_request)
        print(f"✅ Decision: {response['outcome']}")
        print(f"   Approved Amount: ")
        print(f"   Interest Rate: {response.get('interest_rate', 0)*100:.2f}%")
        print(f"   Risk Score: {response['risk_score']:.1f}")
        print(f"   Risk Level: {response['risk_level']}")
        print(f"   Processing Time: {response['processing_time_ms']:.1f}ms")
    except Exception as e:
        print(f"❌ Decision request failed: {e}")
    
    # Test 3: Risk Assessment
    print("\n3️⃣ Risk Assessment Test")
    print("-"*40)
    
    risk_request = {
        "decision_type": "risk_assessment",
        "customer_data": {
            "customer_id": "TEST_002",
            "annual_income": 60000,
            "credit_score": 650,
            "existing_debt": 40000,
            "monthly_debt_payments": 2000,
            "requested_amount": 20000,
            "requested_term_months": 48
        }
    }
    
    try:
        response = client.make_decision(risk_request)
        print(f"✅ Risk Assessment Complete")
        print(f"   Risk Score: {response['risk_score']:.1f}/100")
        print(f"   Risk Level: {response['risk_level']}")
        print(f"   PD: {response.get('probability_of_default', 0):.2%}")
        print(f"   Key Factors: {len(response.get('key_factors', []))} identified")
    except Exception as e:
        print(f"❌ Risk assessment failed: {e}")
    
    # Test 4: Pricing Decision
    print("\n4️⃣ Pricing Decision Test")
    print("-"*40)
    
    pricing_request = {
        "decision_type": "pricing_decision",
        "customer_data": {
            "customer_id": "TEST_003",
            "annual_income": 120000,
            "credit_score": 780,
            "existing_debt": 10000,
            "monthly_debt_payments": 500,
            "requested_amount": 50000,
            "requested_term_months": 36
        }
    }
    
    try:
        response = client.make_decision(pricing_request)
        print(f"✅ Pricing Decision Complete")
        print(f"   Interest Rate: {response.get('interest_rate', 0)*100:.2f}% APR")
        print(f"   Monthly Payment: ")
        
        # Calculate total cost
        if response.get('monthly_payment') and response.get('approved_term'):
            total_paid = response['monthly_payment'] * response['approved_term']
            total_interest = total_paid - response.get('approved_amount', 0)
            print(f"   Total Interest: ")
    except Exception as e:
        print(f"❌ Pricing decision failed: {e}")
    
    # Test 5: Batch Processing
    print("\n5️⃣ Batch Processing Test")
    print("-"*40)
    
    batch_requests = [
        {
            "decision_type": "credit_approval",
            "customer_data": {
                "customer_id": f"BATCH_{i:03d}",
                "annual_income": 50000 + i * 10000,
                "credit_score": 650 + i * 20,
                "existing_debt": 10000 + i * 5000,
                "monthly_debt_payments": 500 + i * 100,
                "requested_amount": 15000 + i * 5000,
                "requested_term_months": 36
            }
        }
        for i in range(3)
    ]
    
    try:
        start_time = time.time()
        responses = client.batch_decision(batch_requests)
        batch_time = (time.time() - start_time) * 1000
        
        print(f"✅ Batch Processing Complete")
        print(f"   Requests Processed: {len(responses)}")
        print(f"   Total Time: {batch_time:.1f}ms")
        print(f"   Average Time: {batch_time/len(responses):.1f}ms per request")
        
        # Summary
        approved = sum(1 for r in responses if r['outcome'] == 'approved')
        print(f"   Approved: {approved}/{len(responses)}")
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
    
    # Test 6: Performance Metrics
    print("\n6️⃣ Performance Metrics")
    print("-"*40)
    
    try:
        metrics = client.get_metrics()
        print(f"✅ Metrics Retrieved")
        print(f"   Total Requests: {metrics['total_requests']}")
        print(f"   Cache Hit Rate: {metrics['cache_hit_rate']:.1%}")
        print(f"   Avg Response Time: {metrics['average_response_time_ms']:.2f}ms")
    except Exception as e:
        print(f"❌ Metrics retrieval failed: {e}")
    
    print("\n" + "="*80)
    print("✅ API Testing Complete!")

if __name__ == "__main__":
    print("💡 Starting API Test Client")
    print("Make sure the API is running on http://localhost:8000")
    print("Press Enter to start tests...")
    input()
    
    run_tests()
