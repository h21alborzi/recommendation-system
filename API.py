from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch

# Load your trained LLaMA 2-based model
# Assume you have a function to generate recommendations
from your_model_module import get_recommendations  

app = FastAPI()

class UserInput(BaseModel):
    user_id: str  # Or any other relevant data
    preferences: List[str]  # Example: ["red", "jeans", "casual"]

@app.post("/recommend/")
async def recommend_items(user_data: UserInput):
    recommendations = get_recommendations(user_data.user_id, user_data.preferences)
    return {"recommendations": recommendations}

