#Detect hallucinations: Measures how much the LLM answer is based on the retrieved context
import numpy as np
from typing import List
import json
import time
from testing.metrics.parse_json_response import _parse_json_response

# Faithfulness prompts
STATEMENT_GENERATION_PROMPT = """
### Task
Break down the answer into atomic statements that are fully understandable without pronouns.
Respond ONLY with a JSON array of strings.

### Example
Question: "Who was Albert Einstein?"
Answer: "He was a German physicist known for relativity."
Output: ["Albert Einstein was a German physicist", "Albert Einstein is known for relativity"]

### Actual Input
Question: "{question}"
Answer: "{answer}"

### Generated Statements:
"""

FAITHFULNESS_EVALUATION_PROMPT = """
### Task
Judge if each statement can be directly inferred from the context. 
Respond ONLY with a JSON array of objects, each containing:
- "statement": the exact statement
- "verdict": 1 (supported) or 0 (not supported)
- "reason": brief explanation (1 sentence)

### Context
{context}

### Statements to Evaluate
{statements}

### Example Response
[
  {{"statement": "John is a computer science major", "verdict": 1, "reason": "Context says John studies Computer Science"}},
  {{"statement": "John works part-time", "verdict": 0, "reason": "No mention of employment in context"}}
]

### Your Response:
"""


def compute_faithfulness(
    question: str,
    answer: str,
    contexts: List[str],
    call_gemini,
    max_retries: int = 5
) -> float:
    """
    Calculate faithfulness score (0.0-1.0) by measuring what percentage of 
    answer statements are supported by the context.
    """
    if not answer.strip():
        return 1.0, 0
    
    context_str = "\n".join(contexts)
    if not context_str.strip():
        return 0.0, 0
    
    total_token_usage = 0
    # Step 1: Generate atomic statements from answer
    prompt = STATEMENT_GENERATION_PROMPT.format(
        question=question,
        answer=answer
    )
    
    statements = []
    for attempt in range(1,max_retries + 1):
        try:
            response, tokens = call_gemini(prompt)
            statements = _parse_json_response(response, [])
            if statements:
                total_token_usage += tokens
                break
        except Exception:
            time.sleep(10*attempt)
            continue
    
    if not statements:
        return np.nan, np.nan
    
    # Step 2: Evaluate statement faithfulness
    eval_prompt = FAITHFULNESS_EVALUATION_PROMPT.format(
        context=context_str,
        statements=json.dumps(statements)
    )
    
    for attempt in range(1,max_retries + 1):
        try:
            response, tokens = call_gemini(eval_prompt)
            verdicts = _parse_json_response(response, [])
            if verdicts:
                valid_verdicts = []
                for v in verdicts:
                    if isinstance(v, dict) and "verdict" in v and v["verdict"] in {0, 1}:
                        valid_verdicts.append(v["verdict"])
                if valid_verdicts:
                    total_token_usage += tokens
                    return sum(valid_verdicts) / len(valid_verdicts), total_token_usage
        except Exception:
            time.sleep(10*attempt)
            continue
    
    return np.nan, np.nan