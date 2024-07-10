from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

app = FastAPI()

model_name_or_path = './results'
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)

class TextPrompt(BaseModel):
    prompt: str

@app.post("/generate/")
async def generate_text(request: TextPrompt):
    inputs = tokenizer.encode(request.prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, temperature=0.7, top_p=0.9, top_k=50)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"generated_text": text}

# Run the server with: uvicorn app:app --reload
