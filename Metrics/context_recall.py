#Evaluate retrieval coverage: Measures how much of the reference (ground-truth) answer is supported by the retrieved context.
from typing import List
import numpy as np
import time
from testing.metrics.parse_json_response import _parse_json_response

# Context Recall prompt
CONTEXT_RECALL_PROMPT = """
### Task
Analyze each sentence in the Answer and determine if it can be attributed to the Context.
Respond ONLY with a JSON object containing a "classifications" list. Each item should have:
- "statement": the exact sentence from Answer
- "reason": brief explanation (1 sentence)
- "attributed": 1 for yes, 0 for no

### Example
Input:
Context: "Einstein won the Nobel Prize in 1921 for physics."
Answer: "Einstein received the Nobel Prize. He was born in Germany."

Output:
{{
  "classifications": [
    {{
      "statement": "Einstein received the Nobel Prize",
      "reason": "Matches context about Nobel Prize",
      "attributed": 1
    }},
    {{
      "statement": "He was born in Germany",
      "reason": "Birth information not in context",
      "attributed": 0
    }}
  ]
}}

### Actual Input
Context: "{context}"

Answer: "{answer}"

Question: "{question}" (for reference only)

### Your Response:
"""

def compute_context_recall(
    question: str,
    contexts: List[str],
    reference_answer: str,
    call_gemini,
    max_retries: int = 5
) -> float:
    """
    Calculate context recall score (0.0-1.0) by measuring what percentage of 
    reference answer statements are supported by the context.
    """
    if not reference_answer.strip():
        return 1.0, 0
    
    context_str = "\n".join(contexts)
    if not context_str.strip():
        return 0.0, 0
    
    prompt = CONTEXT_RECALL_PROMPT.format(
        question=question,
        context=context_str,
        answer=reference_answer
    )
    
    for attempt in range(1, max_retries + 1):
        try:
            response, tokens = call_gemini(prompt)
            data = _parse_json_response(response, {})
            classifications = data.get("classifications", [])
            if classifications:
                attributed = [c.get("attributed", 0) for c in classifications 
                              if isinstance(c, dict) and c.get("attributed") in {0, 1}]
                if attributed:
                    return sum(attributed) / len(attributed), tokens
        except Exception:
            time.sleep(10*attempt)
            continue
    
    return np.nan, np.nan