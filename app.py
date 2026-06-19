from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import time
from model import llama_response, qwen_response, gpt_response

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class GenerateRequest(BaseModel):
    message: str
    model: str

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/generate")
async def generate(req: GenerateRequest):
    user_message = req.message
    model = req.model
    
    if not user_message or not model:
        raise HTTPException(status_code=400, detail="Missing message or model selection")
        
    system_prompt = "You are an AI assistant helping with customer inquiries. Provide a helpful and concise response."
    
    start_time = time.time()
    
    try:
        if model == 'llama':
            result = llama_response(system_prompt, user_message)
        elif model == 'qwen':
            result = qwen_response(system_prompt, user_message)
        elif model == 'gpt':
            result = gpt_response(system_prompt, user_message)
        else:
            raise HTTPException(status_code=400, detail="Invalid model selection")
        
        result['duration'] = time.time() - start_time
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)
