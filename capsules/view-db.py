import sys
sys.path.insert(0, 'src/services/capsule_service')

from models import SessionLocal, Capsule

db = SessionLocal()
capsules = db.query(Capsule).all()

print("\n" + "="*80)
print("CAPSULES DATABASE - PostgreSQL")
print("="*80 + "\n")

for c in capsules:
    print(f"ID: {c.id}")
    print(f"  Client: {c.client_id}")
    print(f"  Type: {c.capsule_type}")
    print(f"  Goal: ${c.goal_amount:,.0f} by {c.goal_date}")
    print(f"  Current Value: ${c.current_value:,.0f}")
    print(f"  Status: {c.status}")
    print(f"  Created: {c.created_at}")
    print()

print(f"Total Capsules: {len(capsules)}")
print("="*80 + "\n")

db.close()
