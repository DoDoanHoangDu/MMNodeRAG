#Weighted score of F-1 factuality score and semantic similarity
import numpy as np
import time
import json
from typing import List
from testing.metrics.parse_json_response import _parse_json_response

# Answer Accuracy prompts
STATEMENT_GENERATOR_PROMPT = """
Generate concise independent statements from the given text that represent factual claims.
Respond ONLY with a JSON array of strings. Do not include any other text.

Example Input: 
"The sun is powered by nuclear fusion. This process creates light and heat."

Example Output:
["The sun is powered by nuclear fusion", "Nuclear fusion creates light and heat"]

Input Text:
{text}

Generated Statements:
"""

CORRECTNESS_EXAMPLES = [
    {
        "input": {
            "question": "What powers the sun and its primary function?",
            "answer": [
                "The sun is powered by nuclear fission",
                "Its primary function is providing light"
            ],
            "ground_truth": [
                "The sun is powered by nuclear fusion",
                "Fusion creates energy for heat and light",
                "Sunlight is essential for Earth's climate"
            ]
        },
        "output": {
            "TP": [{"statement": "Its primary function is providing light", "reason": "Matches ground truth about light"}],
            "FP": [{"statement": "The sun is powered by nuclear fission", "reason": "Contradicts fusion fact"}],
            "FN": [
                {"statement": "The sun is powered by nuclear fusion", "reason": "Missing correct power source"},
                {"statement": "Fusion creates energy for heat and light", "reason": "Missing energy creation detail"}
            ]
        }
    }
]

CORRECTNESS_PROMPT_TEMPLATE = """
Analyze statements from an answer compared to ground truth. Classify each as:
- TP (True Positive): Present in answer and supported by ground truth
- FP (False Positive): Present in answer but unsupported
- FN (False Negative): Missing from answer but present in ground truth

Provide JSON output with lists of TP, FP, FN objects containing 'statement' and 'reason'.


Examples:
{examples}

Current Analysis:
Question: "{question}"
Answer Statements: {answer}
Ground Truth Statements: {ground_truth}

Respond ONLY with valid JSON in this format:
{{"TP": [...], "FP": [...], "FN": [...]}}
"""


def fbeta_score(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    """Calculate F-beta score."""
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-10)


def compute_answer_accuracy(
    question: str,
    answer: str,
    ground_truth: str,
    call_gemini,
    create_embedding,
    weights: List[float] = [0.75, 0.25],
    beta: float = 1.0,
    max_retries: int = 5
) -> float:
    """
    Compute answer correctness score combining factuality and semantic similarity.
    """
    total_token_usage = 0

    # Generate statements from answer
    answer_prompt = STATEMENT_GENERATOR_PROMPT.format(text=answer)
    gt_prompt = STATEMENT_GENERATOR_PROMPT.format(text=ground_truth)
    
    answer_statements = []
    gt_statements = []
    
    for attempt in range(1, max_retries + 1):
        try:
            resp, tokens = call_gemini(answer_prompt)
            answer_statements = _parse_json_response(resp, [])
            if answer_statements:
                total_token_usage += tokens
                break
        except Exception:
            time.sleep(10*attempt)
            continue
    
    for attempt in range(1, max_retries + 1):
        try:
            resp, tokens = call_gemini(gt_prompt)
            gt_statements = _parse_json_response(resp, [])
            if gt_statements:
                total_token_usage += tokens
                break
        except Exception:
            time.sleep(10*attempt)
            continue
    
    # Calculate factuality score
    factuality_score = 0.0
    if weights[0] != 0:
        if not answer_statements and not gt_statements:
            factuality_score = 1.0
        else:
            examples_text = "\n".join(
                f"Input: {json.dumps(ex['input'])}\nOutput: {json.dumps(ex['output'])}"
                for ex in CORRECTNESS_EXAMPLES)
            
            classify_prompt = CORRECTNESS_PROMPT_TEMPLATE.format(
                examples=examples_text,
                question=question,
                answer=json.dumps(answer_statements),
                ground_truth=json.dumps(gt_statements)
            )
            
            for attempt in range(1,max_retries + 1):
                try:
                    resp, tokens = call_gemini(classify_prompt)
                    data = _parse_json_response(resp, {})
                    tp = len(data.get("TP", []))
                    fp = len(data.get("FP", []))
                    fn = len(data.get("FN", []))
                    factuality_score = fbeta_score(tp, fp, fn, beta)
                    total_token_usage += tokens
                    break
                except Exception:
                    time.sleep(10*attempt)
                    continue
    
    # Calculate semantic similarity
    similarity_score = 0.0
    if weights[1] != 0:
        try:
            a_embed = create_embedding(answer)[0]
            gt_embed = create_embedding(ground_truth)[0]
            cosine_sim = np.dot(a_embed, gt_embed) / (
                np.linalg.norm(a_embed) * np.linalg.norm(gt_embed))
            similarity_score = (cosine_sim + 1) / 2
        except Exception:
            pass
    
    return float(np.average([factuality_score, similarity_score], weights=weights)), total_token_usage