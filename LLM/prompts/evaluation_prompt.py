def evaluation_prompt():
    prompt = """
[ROLE]
You are an answer evaluation system that determines whether a model answer is correct based on a list of acceptable answers.

[TASK]
Given a question, a model answer, and a list of acceptable answers:
- Evaluate whether the model answer is correct.
- Compare the model answer against all acceptable answers.
- Return a binary result indicating correctness.

[RULES]
1. Return 1 if the model answer matches any acceptable answer; otherwise return 0.
2. Acceptable answers may be:
   a) Exact values (string or number):
      - Text: match semantically, case-insensitive, ignore minor formatting differences (whitespace, punctuation).
      - Numbers: must be numerically equal.
   b) Numeric ranges in the form:
      { "value": <number>, "range": [min, max] }
      - The model answer is correct if its numeric value lies within the inclusive range [min, max].
3. If the model answer is numeric or can be parsed as a number, convert it before comparison.
4. Ignore minor formatting differences (extra spaces, capitalization).
5. If multiple acceptable answers are provided, matching any one is sufficient.
6. If the answer cannot be interpreted or parsed when required, return 0.

[INPUT FORMAT]
{
  "question": "<string>",
  "model_answer": "<string>",
  "acceptable_answers": [
    "<string or number>",
    { "value": <number>, "range": [min, max] }
  ]
}

[OUTPUT FORMAT]
Return only:
1  (correct)
or
0  (incorrect)

[EXAMPLES]
Input:
{
  "question": "What is 2 + 2?",
  "model_answer": "The result is four.",
  "acceptable_answers": ["4", 4, { "value": 4, "range": [3.9, 4.1] }]
}

Output:
1

--- 
Input:
{
  "question": "Who wrote 'Romeo and Juliet'?",
  "model_answer": "william shakespeare did",
  "acceptable_answers": [
    "William Shakespeare",
    "Shakespeare"
  ]
}

Output:
1
"""
    return prompt.strip()