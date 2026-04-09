def high_level_overview_prompt(text):
    prompt = f"""
-- SYSTEM INSTRUCTION --
You are an expert indexer for a Retrieval-Augmented Generation (RAG) system. Your task is to extract a set of highly relevant keywords that represent the core concepts of the provided high-level summary.

-- OUTPUT REQUIREMENTS
1. The keywords should be concise noun phrases or entities (1-5 words)
2. The keywords should capture distinct aspects of the content
3. The list must stay in the range of 3-50 keywords
4. Return ONLY a valid JSON array of strings
5. Do NOT include any explanation, text, or formatting outside the JSON array

-- USER INPUT --
{text}
"""
    return prompt