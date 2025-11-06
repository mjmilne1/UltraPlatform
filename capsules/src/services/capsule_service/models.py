from sqlalchemy import create_engine, Column, String, Float, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import enum
import os
from dotenv import load_dotenv

load_dotenv()
Base = declarative_base()

# Enums
class TransactionType(str, enum.Enum):
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    DIVIDEND = "dividend"
    INTEREST = "interest"
    FEE = "fee"
    REBALANCE = "rebalance"

class AssetClass(str, enum.Enum):
    CASH = "cash"
    STOCKS = "stocks"
    BONDS = "bonds"
    REAL_ESTATE = "real_estate"
    COMMODITIES = "commodities"
    CRYPTO = "crypto"

# Main Capsule Model
class Capsule(Base):
    __tablename__ = 'capsules'
    
    id = Column(String, primary_key=True)
    client_id = Column(String, nullable=False, index=True)
    capsule_type = Column(String, nullable=False)
    status = Column(String, default='created')
    goal_amount = Column(Float, nullable=False)
    goal_date = Column(String, nullable=False)
    current_value = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    transactions = relationship("Transaction", back_populates="capsule", cascade="all, delete-orphan")
    allocations = relationship("Allocation", back_populates="capsule", cascade="all, delete-orphan")
    performance = relationship("Performance", back_populates="capsule", cascade="all, delete-orphan")
    
    def to_dict(self):
        return {
            'id': self.id,
            'client_id': self.client_id,
            'capsule_type': self.capsule_type,
            'status': self.status,
            'goal_amount': self.goal_amount,
            'goal_date': self.goal_date,
            'current_value': self.current_value,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# Transaction Model
class Transaction(Base):
    __tablename__ = 'transactions'
    
    id = Column(String, primary_key=True)
    capsule_id = Column(String, ForeignKey('capsules.id'), nullable=False)
    transaction_type = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    description = Column(String)
    transaction_date = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    capsule = relationship("Capsule", back_populates="transactions")
    
    def to_dict(self):
        return {
            'id': self.id,
            'capsule_id': self.capsule_id,
            'transaction_type': self.transaction_type,
            'amount': self.amount,
            'description': self.description,
            'transaction_date': self.transaction_date.isoformat() if self.transaction_date else None
        }

# Allocation Model
class Allocation(Base):
    __tablename__ = 'allocations'
    
    id = Column(String, primary_key=True)
    capsule_id = Column(String, ForeignKey('capsules.id'), nullable=False)
    asset_class = Column(String, nullable=False)
    target_percentage = Column(Float, nullable=False)
    current_percentage = Column(Float, default=0.0)
    current_value = Column(Float, default=0.0)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    capsule = relationship("Capsule", back_populates="allocations")
    
    def to_dict(self):
        return {
            'id': self.id,
            'capsule_id': self.capsule_id,
            'asset_class': self.asset_class,
            'target_percentage': self.target_percentage,
            'current_percentage': self.current_percentage,
            'current_value': self.current_value,
            'drift': round(self.current_percentage - self.target_percentage, 2)
        }

# Performance Model
class Performance(Base):
    __tablename__ = 'performance'
    
    id = Column(String, primary_key=True)
    capsule_id = Column(String, ForeignKey('capsules.id'), nullable=False)
    period = Column(String, nullable=False)  # daily, weekly, monthly, yearly
    start_value = Column(Float, nullable=False)
    end_value = Column(Float, nullable=False)
    return_percentage = Column(Float, nullable=False)
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    capsule = relationship("Capsule", back_populates="performance")
    
    def to_dict(self):
        return {
            'id': self.id,
            'capsule_id': self.capsule_id,
            'period': self.period,
            'start_value': self.start_value,
            'end_value': self.end_value,
            'return_percentage': self.return_percentage,
            'period_start': self.period_start.isoformat() if self.period_start else None,
            'period_end': self.period_end.isoformat() if self.period_end else None
        }

# Database setup
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///capsules.db')
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)
    print("? Database tables created (capsules, transactions, allocations, performance)")
