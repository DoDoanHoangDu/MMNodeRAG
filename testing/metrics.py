"""
Metrics using Gemini LLM client instead of LangChain.
All functions are synchronous to match call_gemini.
"""
import json
import numpy as np
from typing import List
from rouge_score import rouge_scorer #type: ignore

import sys
sys.path.insert(0, '.')
from hybrid_graphrag.generation.llm_client import call_gemini
from hybrid_graphrag.knowledge_base.embeddings import init_encoder, create_embedding


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


def _parse_json_response(response: str, default=None):
    """Parse JSON from LLM response, handling markdown code blocks."""
    if default is None:
        default = []
    try:
        # Try direct parse
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code block
    if "```" in response:
        try:
            start = response.find("```")
            end = response.rfind("```")
            if start != end:
                code = response[start:end]
                # Remove or ``` prefix
                if code.startswith(""):
                    code = code[7:]
                elif code.startswith("```"):
                    code = code[3:]
                return json.loads(code.strip())
        except json.JSONDecodeError:
            pass
    
    # Try finding JSON array or object
    for start_char, end_char in [('[', ']'), ('{', '}')]:
        start = response.find(start_char)
        end = response.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(response[start:end+1])
            except json.JSONDecodeError:
                continue
    
    return default


def compute_faithfulness(
    question: str,
    answer: str,
    contexts: List[str],
    max_retries: int = 2
) -> float:
    """
    Calculate faithfulness score (0.0-1.0) by measuring what percentage of 
    answer statements are supported by the context.
    """
    if not answer.strip():
        return 1.0
    
    context_str = "\n".join(contexts)
    if not context_str.strip():
        return 0.0
    
    # Step 1: Generate atomic statements from answer
    prompt = STATEMENT_GENERATION_PROMPT.format(
        question=question[:500],
        answer=answer[:3000]
    )
    
    statements = []
    for _ in range(max_retries + 1):
        try:
            response = call_gemini(prompt, "You are a helpful assistant.")
            statements = _parse_json_response(response, [])
            if statements:
                break
        except Exception:
            continue
    
    if not statements:
        return np.nan
    
    # Step 2: Evaluate statement faithfulness
    eval_prompt = FAITHFULNESS_EVALUATION_PROMPT.format(
        context=context_str[:10000],
        statements=json.dumps(statements)[:5000]
    )
    
    for _ in range(max_retries + 1):
        try:
            response = call_gemini(eval_prompt, "You are a helpful assistant.")
            verdicts = _parse_json_response(response, [])
            if verdicts:
                valid_verdicts = []
                for v in verdicts:
                    if isinstance(v, dict) and "verdict" in v and v["verdict"] in {0, 1}:
                        valid_verdicts.append(v["verdict"])
                if valid_verdicts:
                    return sum(valid_verdicts) / len(valid_verdicts)
        except Exception:
            continue
    
    return np.nan


def compute_context_recall(
    question: str,
    contexts: List[str],
    reference_answer: str,
    max_retries: int = 2
) -> float:
    """
    Calculate context recall score (0.0-1.0) by measuring what percentage of 
    reference answer statements are supported by the context.
    """
    if not reference_answer.strip():
        return 1.0
    
    context_str = "\n".join(contexts)
    if not context_str.strip():
        return 0.0
    
    prompt = CONTEXT_RECALL_PROMPT.format(
        question=question,
        context=context_str[:10000],
        answer=reference_answer[:2000]
    )
    
    for _ in range(max_retries + 1):
        try:
            response = call_gemini(prompt, "You are a helpful assistant.")
            data = _parse_json_response(response, {})
            classifications = data.get("classifications", [])
            if classifications:
                attributed = [c.get("attributed", 0) for c in classifications 
                              if isinstance(c, dict) and c.get("attributed") in {0, 1}]
                if attributed:
                    return sum(attributed) / len(attributed)
        except Exception:
            continue
    
    return np.nan


def compute_context_relevance(
    question: str,
    contexts: List[str],
    max_retries: int = 3
) -> float:
    """
    Evaluate the relevance of retrieved contexts for answering a question.
    Returns a score between 0.0 (irrelevant) and 1.0 (fully relevant).
    """
    if not question.strip() or not contexts or not any(c.strip() for c in contexts):
        return 0.0
    
    context_str = "\n".join(contexts)[:7000]
    
    if context_str.strip() == question.strip():
        return 0.0
    
    prompt = CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context_str)
    
    ratings = []
    for _ in range(2):  # Get two ratings
        for _ in range(max_retries):
            try:
                response = call_gemini(prompt, "You are a helpful assistant.")
                # Parse rating from response
                for token in response.split()[:8]:
                    if token.isdigit() and 0 <= int(token) <= 2:
                        ratings.append(float(token) / 2)
                        break
                if len(ratings) > 0:
                    break
            except Exception:
                continue
    
    if not ratings:
        return np.nan
    return sum(ratings) / len(ratings)


def compute_coverage(
    question: str,
    reference: str,
    response: str,
    max_retries: int = 1
) -> float:
    """
    Calculate coverage score (0.0-1.0) by measuring what percentage of 
    reference facts are covered in the response.
    """
    if not reference.strip():
        return 1.0
    
    # Step 1: Extract facts from reference
    extract_prompt = FACT_EXTRACTION_PROMPT.format(
        question=question,
        reference=reference[:3000]
    )
    
    facts = []
    for _ in range(max_retries + 1):
        try:
            resp = call_gemini(extract_prompt, "You are a helpful assistant.")
            data = _parse_json_response(resp, {})
            facts = [str(f) for f in data.get("facts", []) if f]
            if facts:
                break
        except Exception:
            continue
    
    if not facts:
        return np.nan
    
    # Step 2: Check fact coverage
    coverage_prompt = FACT_COVERAGE_PROMPT.format(
        question=question,
        response=response[:3000],
        facts=json.dumps(facts)
    )
    
    for _ in range(max_retries + 1):
        try:
            resp = call_gemini(coverage_prompt, "You are a helpful assistant.")
            data = _parse_json_response(resp, {})
            classifications = data.get("classifications", [])
            if classifications:
                attributed = [c.get("attributed", 0) for c in classifications 
                              if isinstance(c, dict) and c.get("attributed") in {0, 1}]
                if attributed:
                    return sum(attributed) / len(attributed)
        except Exception:
            continue
    
    return np.nan


def fbeta_score(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    """Calculate F-beta score."""
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-10)


def compute_answer_accuracy(
    question: str,
    answer: str,
    ground_truth: str,
    embedding_model=None,
    weights: List[float] = [0.75, 0.25],
    beta: float = 1.0,
    max_retries: int = 1
) -> float:
    """
    Compute answer correctness score combining factuality and semantic similarity.
    """
    # Generate statements from answer
    answer_prompt = STATEMENT_GENERATOR_PROMPT.format(text=answer)
    gt_prompt = STATEMENT_GENERATOR_PROMPT.format(text=ground_truth)
    
    answer_statements = []
    gt_statements = []
    
    for _ in range(max_retries + 1):
        try:
            resp = call_gemini(answer_prompt, "You are a helpful assistant.")
            answer_statements = _parse_json_response(resp, [])
            if answer_statements:
                break
        except Exception:
            continue
    
    for _ in range(max_retries + 1):
        try:
            resp = call_gemini(gt_prompt, "You are a helpful assistant.")
            gt_statements = _parse_json_response(resp, [])
            if gt_statements:
                break
        except Exception:
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
            
            for _ in range(max_retries + 1):
                try:
                    resp = call_gemini(classify_prompt, "You are a helpful assistant.")
                    data = _parse_json_response(resp, {})
                    tp = len(data.get("TP", []))
                    fp = len(data.get("FP", []))
                    fn = len(data.get("FN", []))
                    factuality_score = fbeta_score(tp, fp, fn, beta)
                    break
                except Exception:
                    continue
    
    # Calculate semantic similarity
    similarity_score = 0.0
    if weights[1] != 0 and embedding_model is not None:
        try:
            a_embed = create_embedding(embedding_model, answer)[0]
            gt_embed = create_embedding(embedding_model, ground_truth)[0]
            cosine_sim = np.dot(a_embed, gt_embed) / (
                np.linalg.norm(a_embed) * np.linalg.norm(gt_embed))
            similarity_score = (cosine_sim + 1) / 2
        except Exception:
            pass
    
    return float(np.average([factuality_score, similarity_score], weights=weights))


def compute_rouge(
    answer: str,
    ground_truth: str,
    rouge_type: str = "rougeL",
    mode: str = "fmeasure"
    ) -> float:
    """
    Compute ROUGE score between generated answer and ground truth reference.
    """
    if not ground_truth.strip() or not answer.strip():
        return 0.0
    
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    scores = scorer.score(ground_truth, answer)
    return getattr(scores[rouge_type], mode)



