def high_level_overview_prompt(text):
    prompt = f"""
-- SYSTEM INSTRUCTION --
You are an expert indexer for a Retrieval-Augmented Generation (RAG) system. Your sole task is to generate a concise, keyword-rich title for the provided high-level summary. The title must function as a stand-alone keyword search entry point.

Your output must be ONLY the title. Do not include any introductory phrases, explanations, quotation marks, or markdown formatting.

-- USER INPUT --
Generate the High-level Overview keyword title for the following High-level Element summary:
{text}
"""
    return prompt