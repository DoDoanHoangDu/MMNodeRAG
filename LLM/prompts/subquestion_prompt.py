def subquestion_prompt(question):
    return f"""
[ROLE]
You are a question decomposition module.

[INPUT FORMAT] 
An image and a question about the image.

[REAL INPUT]
{question}

[TASK]
Generate the prerequisite sub-questions needed to answer the question.

[RULES]
1. Each sub-question must be:
   - Atomic (cannot be split further)
   - Specific and self-contained

2. Only include prerequisite questions:
   - These are questions that needed to be answered BEFORE the original question
   - Do NOT include the original question or any rewording of it

3. Decomposition strategy:
   - Identify the primary entities in the image if they are relevant to the text query

4. Keep the number of sub-questions between 0 and 4.

5. Avoid redundancy:
   - Do not ask overlapping or duplicate questions

6. Do NOT:
   - Answer the questions
   - Explain your reasoning
   - Add any text outside the required format

[OUTPUT FORMAT (STRICT)]
Return ONLY a JSON array of strings.

[EXAMPLE]

Input:
"What country is the brand of this car from?"

Output:
[
  "What brand is shown on the car in the image?"
]

---

Input:
"What is the population of the city shown in the image?"

Output:
[
  "Which city is shown in the image?"
]
"""