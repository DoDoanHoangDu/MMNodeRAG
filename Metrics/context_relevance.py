#Evaluate the relevance of the Context for answering the Question using ONLY the information provided: score 0-2
from typing import List
import numpy as np
import time
# Context Relevance prompt
CONTEXT_RELEVANCE_PROMPT = """
### Task
Evaluate the relevance of the Context for answering the Question using ONLY the information provided.
Respond ONLY with a number from 0-2. Do not explain.

### Rating Scale
0: Context has NO relevant information
1: Context has PARTIAL relevance
2: Context has RELEVANT information

### Question
{question}

### Context
{context}

### Rating:
"""

def compute_context_relevance(
    question: str,
    contexts: List[str],
    call_gemini,
    max_retries: int = 5
) -> float:
    """
    Evaluate the relevance of retrieved contexts for answering a question.
    Returns a score between 0.0 (irrelevant) and 1.0 (fully relevant).
    """
    if not question.strip() or not contexts or not any(c.strip() for c in contexts):
        return 0.0, 0
    
    context_str = "\n".join(contexts)[:7000]
    
    if context_str.strip() == question.strip():
        return 0.0, 0
    
    prompt = CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context_str)
    
    total_token_usage = 0
    ratings = []
    for _ in range(2):  # Get two ratings
        for attempt in range(1, max_retries+1):
            try:
                response, tokens = call_gemini(prompt)
                # Parse rating from response
                for token in response.split()[:8]:
                    if token.isdigit() and 0 <= int(token) <= 2:
                        ratings.append(float(token) / 2)
                        break
                if len(ratings) > 0:
                    total_token_usage += tokens
                    break
            except Exception:
                time.sleep(10*attempt)
                continue
    
    if not ratings:
        return np.nan, np.nan
    return sum(ratings) / len(ratings), total_token_usage