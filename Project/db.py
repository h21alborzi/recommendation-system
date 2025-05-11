from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

Base = declarative_base()
engine = create_engine('sqlite:///search_history.db', connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)

class SearchHistory(Base):
    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    query = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    results_json = Column(Text)

Base.metadata.create_all(bind=engine)

def log_search(user_id, query, results):
    session = SessionLocal()
    log = SearchHistory(
        user_id=user_id,
        query=query,
        results_json=json.dumps(results)
    )
    session.add(log)
    session.commit()
    session.close()
