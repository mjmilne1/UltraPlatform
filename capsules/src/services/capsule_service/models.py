from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
Base = declarative_base()

class Capsule(Base):
    __tablename__ = 'capsules'
    id = Column(String, primary_key=True)
    client_id = Column(String, nullable=False)
    capsule_type = Column(String, nullable=False)
    status = Column(String, default='created')
    goal_amount = Column(Float, nullable=False)
    goal_date = Column(String, nullable=False)
    current_value = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id, 'client_id': self.client_id,
            'capsule_type': self.capsule_type, 'status': self.status,
            'goal_amount': self.goal_amount, 'goal_date': self.goal_date,
            'current_value': self.current_value,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/capsules')
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
    print("✓ Tables created")
