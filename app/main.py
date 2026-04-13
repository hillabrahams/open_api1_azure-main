from fastapi import FastAPI, HTTPException
from app.models import EntryRequest, EntryAnalysis
from app.openai_service import analyze_entry

app = FastAPI()

@app.post("/analyze/", response_model=EntryAnalysis)
async def analyze_journal_entry(request: EntryRequest):
    try:
        analysis = analyze_entry(request.entry)
        analysis["affect"] = request.affect if request.affect is not None else 0
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
