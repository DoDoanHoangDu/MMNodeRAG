#Evaluate retrieval coverage: Measures how much of the reference (ground-truth) answer is supported by the retrieved context.
from typing import List
import numpy as np
import time
from Metrics.parse_json_response import _parse_json_response
from LLM.call_api import call_api

# Context Recall prompt
SYSTEM_PROMPT = """
[TASK]
Analyze each sentence in the Answer and determine if it can be attributed to the Context.
Respond ONLY with a JSON object containing a "classifications" list. Each item should have:
- "statement": the exact sentence from Answer
- "reason": brief explanation (1 sentence)
- "attributed": 1 for yes, 0 for no

[CRITICAL RULES]
- Split the Answer into sentences.
- If the Answer has only one sentence OR cannot be clearly split, treat the entire Answer as ONE statement.
- You MUST return at least ONE classification.

[EXAMPLE]
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
"""

USER_PROMPT = """
[USER INPUT]
Question: "{question}" (for reference only)

Answer: 
"{answer}"

Context: 
"{context}"
"""

def compute_context_recall(
    question: str,
    contexts: List[str],
    reference_answer: str,
    max_retries: int = 10
) -> float:
    """
    Calculate context recall score (0.0-1.0) by measuring what percentage of 
    reference answer statements are supported by the context.
    """
    if not reference_answer.strip():
        return 1.0, 0
    
    context_str = "\n------\n".join(contexts)
    if not context_str.strip():
        return 0.0, 0
    
    prompt = USER_PROMPT.format(
        question=question,
        context=context_str,
        answer=reference_answer
    )
    
    for attempt in range(1, max_retries + 1):
        try:
            response, tokens = call_api(content = [{"type": "text", "text": prompt}], system_prompt=SYSTEM_PROMPT, mode = "self-host")
            data = _parse_json_response(response, {})
            classifications = data.get("classifications", [])
            if classifications:
                attributed = [c.get("attributed", 0) for c in classifications 
                              if isinstance(c, dict) and c.get("attributed") in {0, 1}]
                if attributed:
                    return sum(attributed) / len(attributed), tokens
            print(question)
            print(reference_answer)
            print(f"Warning: No valid classifications found in response. Attempt {attempt}/{max_retries}. Response: {response}")
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(f"Failed to compute context recall after {max_retries} attempts")
            #time.sleep(10*attempt)
    
    return np.nan, np.nan