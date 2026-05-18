def entity_matching_prompt(entities):
    SYSTEM_PROMPT = rf"""
[GOAL]
Given a list of entities, group them into clusters of synonyms.

[DEFINITION]
Two entities are considered synonyms if they refer to the same concept, object, or meaning, even if phrased differently. This includes:
- lexical variants (e.g., "car" vs "automobile")
- morphological variants (e.g., "cats" vs "cat")
- common abbreviations (e.g., "USA" vs "United States of America", "NYC" vs "New York City")
- common paraphrases

Do NOT group entities that are only loosely related or belong to the same category but have different meanings.

Specifically:
- Do NOT group entities that have a hierarchical relationship (e.g., general category vs specific subtype).
- A more specific entity (hyponym) is NOT a synonym of a more general entity (hypernym).

Examples of NON-synonyms:
- "lung cancer" and "cancer" (specific type vs general disease)
- "electric car" and "car" (subcategory vs category)
- "golden retriever" and "dog" (specific breed vs general animal)

Only group entities if they refer to the exact same concept at the same level of specificity.

[STEPS]
1. Analyze the semantic meaning of each entity.
2. Group entities that are true synonyms into the same cluster.
3. Ensure each entity appears in exactly one cluster.
4. If an entity has no synonyms in the list, place it in its own singleton cluster.

[OUTPUT FORMAT]
- Output a list of lists.
- Each inner list represents a cluster of synonyms.
- Preserve the original entity text exactly as given (no modification, no normalization).
- Do not include explanations.

[EXAMPLE]
Input:
["car", "fish", "cat", "feline", "automobile"]

Output:
[["car", "automobile"], ["cat", "feline"], ["fish"]]
"""
    USER_PROMPT = f"[USER INPUT]\n{entities}"
    return SYSTEM_PROMPT, USER_PROMPT