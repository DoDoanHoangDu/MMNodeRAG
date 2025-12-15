def question_decompose_prompt(query):
    prompt = f"""--Goal--
Please break down the following query into a single list.
--Requirement--
1. Each item in the list should either be a main entity (such as a key noun or object).
2. If you have high confidence about the user's intent or domain knowledge, you may also include closely related terms.
    Limit related terms strictly to:
    - Direct synonyms or standard abbreviations.
    - Grammatical variations needed for coverage (e.g., if "prevent" is present, include "prevention").
3. If uncertain, please only extract entities and semantic chunks directly from the query. Please try to
reduce the number of common nouns in the list. Ensure all elements are organized within one unified list.
4. **Return the output strictly as a JSON array string.** 
   Do NOT include any Markdown, explanation, or text outside the JSON list.
--Input--
Query: {query}
"""
    return prompt