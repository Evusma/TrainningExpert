import json
import torch
import configparser

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


config = configparser.ConfigParser()
config.read("config.ini")

config1 = dict(config["config1"])

# Convert dtype safely
dtype_map = {
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.bfloat16": torch.bfloat16,
}

max_new_tokens=int(config1['max_new_tokens']),
use_cache=config1.getboolean('use_cache')

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(config1['model_name'])
model = AutoModelForCausalLM.from_pretrained(
    config1['model_name'],
    dtype=dtype_map[config1["dtype"]],
    device_map=config1['device_map'],
)


class ChatRequest(BaseModel):
    message: str
    tables: dict


class ChatResponse(BaseModel):
    response: str


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```python"):
        text = text[len("```python"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
@torch.inference_mode()
def chat(payload: ChatRequest) -> ChatResponse:
    messages = [
        {
            "role": "system",
            "content": (
                f"{config1['content_system_prompt']} "
                f"{json.dumps(payload.tables, ensure_ascii=False)}"
            )
        },
        {
            "role": "user",
            "content": payload.message,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=use_cache,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return ChatResponse(response=strip_code_fence(response))