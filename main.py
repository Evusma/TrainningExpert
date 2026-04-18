import json
import torch
import configparser

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer


config = configparser.ConfigParser()
config.read("config.ini")

config_file = config["config3"]

# Convert dtype safely
dtype_map = {
    "torch.float16": torch.float16,
    "torch.float32": torch.float32,
    "torch.bfloat16": torch.bfloat16,
}

max_new_tokens=int(config_file['max_new_tokens']),
use_cache=config_file.getboolean('use_cache')

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(config_file['model_name'])
model = AutoModelForCausalLM.from_pretrained(
    config_file['model_name'],
    torch_dtype=dtype_map[config_file["dtype"]],
    device_map=config_file['device_map'],
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
def chat(payload: ChatRequest) -> ChatResponse:
    messages = [
        {
            "role": "system",
            "content": (
                f"{config_file['content_system_prompt']} "
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

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(config_file["max_new_tokens"]),
            do_sample=config_file.getboolean("do_sample"),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=config_file.getboolean("use_cache"),
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return ChatResponse(response=strip_code_fence(response))