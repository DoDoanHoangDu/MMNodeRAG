import os
from openai import OpenAI
from google import genai
from dotenv import load_dotenv
load_dotenv()

openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_KEY")
)

gemini_client = genai.Client(api_key = os.getenv("GEMINI_KEY"))

vastai_client = OpenAI(
        base_url="http://198.53.64.194:35041/v1",
        api_key="EMPTY"
    )

def call_api(content, system_prompt = None, model="qwen/qwen3.6-plus:free", mode = "openai", thinking = False):
    if mode == "openai":
        response = openai_client.chat.completions.create(
            model= model,
            messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                    ],
            extra_body={"reasoning": {"enabled": thinking}}
        )
        return response.choices[0].message.content, response.usage.total_tokens
    elif mode == "gemini":
        thinking_level = 0 if thinking else 512 
        response = gemini_client.models.generate_content(
            model= model,
            contents= content,
            config= genai.types.GenerateContentConfig(
                thinking_config = genai.types.ThinkingConfig(thinkingBudget  = thinking_level)
            )
        )
        return response.text, response.usage_metadata.total_token_count
    elif mode == "self-host":
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append({"role": "user", "content": content})
        response = vastai_client.chat.completions.create(
            model="Qwen/Qwen3-VL-8B-Instruct",
            messages= messages
        )

        return response.choices[0].message.content, response.usage.total_tokens
    else:
        raise ValueError("Unsupported API type.")