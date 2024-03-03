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
    sentiment = Get_sentiment()
    return {"sentiment": sentiment}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
