def question_decompose_prompt(query):
    prompt = f"""
--Role--
You are an expert multimodal information extraction system.

--Goal--
Given:
1. A user question (text)
2. An associated image

Extract a single unified list of key entities relevant to answering the question.

--Definition of "Entity"--
Entities include:
- Objects (physical items)
- People, animals, locations
- Brands, products, or named instances
- Date or time periods
- Essential abstract concepts

--Core Constraints--

1. **Text Grounding (Primary)**
   - Extract ALL entities explicitly present in the text query.
   - Include nouns, noun phrases, and key concepts.

2. **Image Grounding (Conditional)**
   - Entities from the image may ONLY be included if they are clearly relevant to the query.
   - Do NOT include objects that are visible but irrelevant to the question.

3. **Relevance Filtering**
   - Avoid generic or overly broad terms (e.g., "thing", "object", "image", "place", "time").

4. **Controlled Expansion (High Confidence ONLY)**
   If you have high confidence about the user's intent or domain knowledge, you may include closely related terms.

   Limit strictly to:
   - Direct synonyms (e.g., "car", "automobile")
   - Standard abbreviations (e.g., "USA", "United States", "The U.S.")
   - Minimal grammatical variations required for coverage (e.g., "prevent" → "prevention", "birds" → "bird")

5. **Multimodal Consistency Rule**
   - Every entity must be supported by either explicit text in the query OR clear visual evidence that is directly relevant to the query

6. **Output Format (STRICT)**
   - Return ONLY a JSON array of strings
   - No explanation, no markdown, no extra text

--Input--
Question: {query}
"""
    return prompt