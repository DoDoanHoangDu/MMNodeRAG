import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("API_KEY")
)

def call_api(content, model="qwen/qwen3.6-plus:free"):
    response = client.chat.completions.create(
    model= model,
    messages=[
            {
                "role": "user",
                "content": content
            }
            ],
    extra_body={"reasoning": {"enabled": False}}
    )
    return response.choices[0].message.content, response.usage.total_tokens
