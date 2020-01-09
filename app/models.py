from sqlalchemy import Integer, ForeignKey, String, Column, Float, Boolean
from datetime import datetime
from app import db  

'''
Flask Migration Process:
1. (just initially) flask db init (initializes db)
2. (after each change) flask db migrate -m "message" (creates migration)
3. flask db upgrade (performs migration) 
'''

class Algo(db.Model):
    __tablename__ = "algos"
    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String)
    description = Column(String, nullable=True)
    optimizer = Column(String)
    layers = Column(Integer)
    type = Column(String)

    def to_json(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'type': self.type,
            'optimizer': self.optimizer,
            'layers': self.layers
        }

