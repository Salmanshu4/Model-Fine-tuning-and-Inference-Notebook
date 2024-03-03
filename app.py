from fastapi import FastAPI, HTTPException, File, UploadFile
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
import pandas as pd
import torch
from model import Get_sentiment
from transformers import BertTokenizer, TFBertForSequenceClassification

# Initialize FastAPI app
app = FastAPI()


path = '/Users/genericname/Desktop/Dev/sentiment/'
# Load tokenizer
bert_tokenizer = BertTokenizer.from_pretrained(path +'/Tokenizer')
 
# Load model
bert_model = TFBertForSequenceClassification.from_pretrained(path +'/Model')
# Define SQLAlchemy database
SQLALCHEMY_DATABASE_URL = "sqlite:///./sentiment_analysis.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Define SQLAlchemy model
class SentimentAnalysis(Base):
    __tablename__ = "sentiment_analysis"

    id = Column(Integer, primary_key=True, index=True)
    comment_id = Column(Integer, index=True)
    campaign_id = Column(Integer)
    description = Column(String)
    sentiment = Column(String)

# Create database tables
Base.metadata.create_all(bind=engine)


# Pydantic models for request and response
class SentimentInput(BaseModel):
    text: str

class SentimentOutput(BaseModel):
    sentiment: str

# API endpoints
@app.post("/predict", response_model=SentimentOutput)
async def predict_sentiment(sentiment_input: SentimentInput):
    sentiment = Get_sentiment(sentiment_input, Tokenizer, Model)
    return {"sentiment": sentiment}

@app.post("/insert")
async def insert_record(comment_id: int, campaign_id: int, description: str, sentiment: str):
    db = SessionLocal()
    db_record = SentimentAnalysis(comment_id=comment_id, campaign_id=campaign_id, description=description, sentiment=sentiment)
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    db.close()
    return {"message": "Record inserted successfully"}

@app.delete("/delete")
async def delete_record(comment_id: int):
    db = SessionLocal()
    db_record = db.query(SentimentAnalysis).filter(SentimentAnalysis.comment_id == comment_id).first()
    if db_record:
        db.delete(db_record)
        db.commit()
        db.close()
        return {"message": "Record deleted successfully"}
    else:
        db.close()
        raise HTTPException(status_code=404, detail="Record not found")
    
@app.put("/update")
async def update_record(comment_id: int, campaign_id: int, description: str, sentiment: str):
    db = SessionLocal()
    db_record = db.query(SentimentAnalysis).filter(SentimentAnalysis.comment_id == comment_id).first()
    if db_record:
        db_record.campaign_id = campaign_id
        db_record.description = description
        db_record.sentiment = sentiment
        db.commit()
        db.close()
        return {"message": "Record updated successfully"}
    else:
        db.close()
        raise HTTPException(status_code=404, detail="Record not found")

@app.post("/bulk_insert")
async def bulk_insert(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(contents)
    db = SessionLocal()
    for _, row in df.iterrows():
        sentiment = Get_sentiment(row['description'], Tokenizer, Model)
        db_record = SentimentAnalysis(comment_id=row['comment_id'], campaign_id=row['campaign_id'], description=row['description'], sentiment=sentiment)
        db.add(db_record)
    db.commit()
    db.close()
    return {"message": "Bulk insertion successful"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
