import os
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from config import logging

logger = logging.getLogger(__name__)

MaxRetries = 10
Delay = 15

def retry_on_failure(max_retries: int = 3, delay: int = 1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    logger.error(f"Attempt {retries}/{max_retries} failed: {e}")
                    time.sleep(delay)
            raise Exception(f"Failed after {max_retries} attempts")
        return wrapper
    return decorator


def calculate_token_price(prompt_tokens, completion_tokens):
    return 7.13 * (0.01 * prompt_tokens / 1000 + 0.03 * completion_tokens / 1000)

@retry_on_failure(max_retries= MaxRetries, delay=Delay)
def get_rsp_from_GPT(sys_prompt: str, user_prompt: str, model_name: str = 'gpt-4o'):
    """
    Description: 
        Get copywriter from GPT-4 with streaming capability, yielding results in real-time
    Args:
        sys_prompt (str): System prompt
        user_prompt (str): User prompt
    Yields:
        rsp (str): The generated response content in chunks
    """
    client = OpenAI(base_url=os.environ.get('OPENAI_BASE_URL'), api_key=os.environ.get('OPENAI_API_KEY'))
    messages = [{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': user_prompt}]
    
    # Enable streaming
    response = client.chat.completions.create(messages=messages, model=model_name, stream=True)
    
    # Stream through the response and yield chunks
    for chunk in response:
        # Accessing the choices directly from the chunk
        choice = chunk.choices[0]
        if choice.delta.content:
            yield choice.delta.content  # Yield each chunk content

def unit_test():
    sys_prompt = "You are a helpful assistant."
    user_prompt = "你好啊,请问0.9铝合金的密度是多少？"
    rsps, cost = get_rsp_from_GPT(sys_prompt, user_prompt)
    print(rsps, cost)

# python -m LLMClient.llm_call
if __name__ == "__main__":
    unit_test()