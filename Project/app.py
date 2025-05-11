from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_utils import load_model_and_index, search_products
from db import log_search
import uvicorn

from db import SessionLocal, SearchHistory
from fastapi.responses import JSONResponse

class SearchRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = 5

model, index, product_data = load_model_and_index()
app = FastAPI()

@app.post("/search")
def semantic_search(req: SearchRequest):
    try:
        results = search_products(req.query, model, index, product_data, top_k=req.top_k)
        log_search(req.user_id, req.query, results)
        return {"user_id": req.user_id, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


@app.get("/history/{user_id}")
def get_user_history(user_id: str):
    session = SessionLocal()
    history = session.query(SearchHistory).filter(SearchHistory.user_id == user_id).all()
    session.close()
    return JSONResponse(content=[{
        "timestamp": h.timestamp.isoformat(),
        "query": h.query,
        "results": h.results_json
    } for h in history])
