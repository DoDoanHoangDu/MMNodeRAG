#Measures how completely a model’s response covers the factual content of a reference (ground-truth) answer.
import numpy as np
import json
import time
from testing.metrics.parse_json_response import _parse_json_response
# Coverage prompts
FACT_EXTRACTION_PROMPT = """
### Task
Extract distinct factual statements from the reference answer that could be independently verified.
Respond ONLY with a JSON object containing a "facts" list of strings.

### Example
Input:
  Question: "What causes seasons?"
  Reference: "Seasonal changes result from Earth's axial tilt. This tilt causes different hemispheres to receive varying sunlight."

Output:
{{
  "facts": [
    "Seasonal changes result from Earth's axial tilt",
    "The axial tilt causes different hemispheres to receive varying sunlight"
  ]
}}

### Actual Input
Question: "{question}"
Reference Answer: "{reference}"

### Your Response:
"""

FACT_COVERAGE_PROMPT = """
### Task
For each factual statement from the reference, determine if it's covered in the response.
Respond ONLY with a JSON object containing a "classifications" list. Each item should have:
- "statement": the exact fact from reference
- "attributed": 1 if covered, 0 if not

### Example
Response: "Seasons are caused by Earth's tilted axis"
Reference Facts: [
  "Seasonal changes result from Earth's axial tilt",
  "The axial tilt causes different hemispheres to receive varying sunlight"
]

Output:
{{
  "classifications": [
    {{"statement": "Seasonal changes result from Earth's axial tilt", "attributed": 1}},
    {{"statement": "The axial tilt causes different hemispheres to receive varying sunlight", "attributed": 0}}
  ]
}}

### Actual Input
Question: "{question}"
Response: "{response}"
Reference Facts: {facts}

### Your Response:
"""

def compute_coverage(
    question: str,
    reference: str,
    response: str,
    call_gemini,
    max_retries: int = 5
) -> float:
    """
    Calculate coverage score (0.0-1.0) by measuring what percentage of 
    reference facts are covered in the response.
    """
    if not reference.strip():
        return 1.0, 0
    
    # Step 1: Extract facts from reference
    extract_prompt = FACT_EXTRACTION_PROMPT.format(
        question=question,
        reference=reference[:3000]
    )
    
    total_token_usage = 0
    facts = []
    for attempt in range(1,max_retries + 1):
        try:
            resp, tokens = call_gemini(extract_prompt)
            data = _parse_json_response(resp, {})
            facts = [str(f) for f in data.get("facts", []) if f]
            if facts:
                total_token_usage += tokens
                break
        except Exception:
            time.sleep(10 * attempt)
            continue
    
    if not facts:
        return np.nan, np.nan
    
    # Step 2: Check fact coverage
    coverage_prompt = FACT_COVERAGE_PROMPT.format(
        question=question,
        response=response[:3000],
        facts=json.dumps(facts)
    )
    
    for attempt in range(1, max_retries + 1):
        try:
            resp, tokens = call_gemini(coverage_prompt)
            data = _parse_json_response(resp, {})
            classifications = data.get("classifications", [])
            if classifications:
                attributed = [c.get("attributed", 0) for c in classifications 
                              if isinstance(c, dict) and c.get("attributed") in {0, 1}]
                if attributed:
                    total_token_usage += tokens
                    return sum(attributed) / len(attributed), total_token_usage
        except Exception:
            time.sleep(10 * attempt)
            continue
    
    return np.nan, np.nan