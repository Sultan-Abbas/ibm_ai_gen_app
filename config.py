import os
from dotenv import load_dotenv

load_dotenv()

# Groq credentials
GROQ_API_KEY = os.environ.get("GROQ_API", "")

# Model IDs
LLAMA_MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"
QWEN_MODEL_ID = "qwen/qwen3.6-27b"
GPT_MODEL_ID = "openai/gpt-oss-120b"

