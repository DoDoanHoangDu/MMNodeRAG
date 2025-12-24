def entity_matching_prompt(entity, relevant_entities):
    prompt = f"""
-- SYSTEM INSTRUCTION --
Task: Return a list from LIST_OF_ENTITIES IMMEDIATELY related to ORIGINAL. 
Output: Exactly one line (valid JSON-style list). No commentary. Empty list `[]` if no match.

Rules (apply in order):
1. Exact Match: Case-insensitive, ignore whitespace.
2. Suffix-Agnostic: Match after removing "inc", "llc", "ltd", "co", "corp", "corporation", "company", "plc" (and dots). 
3. Acronyms: Match if initials of one equal the other (e.g., "WHO" = "World Health Organization").
4. Orthographic & Grammatical Var: Match small character variations, formatting, plurals, or possessives (e.g., "U.S.A." vs "USA", "O'Neil" vs "Oneil", "Barclay" vs "Barclays").
5. Explicit Alias: Direct links via "aka", "formerly", "/", "—".
6. Semantic Equivalence: Match if the candidate is a globally recognized synonym, brand rename, or parent company. Use conservatively.

Selection Constraints:
- Use only provided LIST_OF_ENTITIES; no external knowledge.
- Maintain original order and exact string formatting.
- No duplicates (keep first occurrence only).

Example 1:
ORIGINAL = "IBM"
LIST_OF_ENTITIES = ["International Business Machines", "I.B.M.", "IBMer", "Microsoft"]
Output:
["International Business Machines", "I.B.M."]

Example 2:
ORIGINAL = "World Health Organization"
LIST_OF_ENTITIES = ["WHO", "World Health Org.", "Health Department", "World Health"]
Output:
["WHO", "World Health Org.", "World Health"]

Example 3 (no match):
ORIGINAL = "Acme Corp"
LIST_OF_ENTITIES = ["Beta LLC", "Gamma Inc"]
Output:
[]

-- USER INPUT --
ORIGINAL = "{entity}"
LIST_OF_ENTITIES = {relevant_entities}
"""
    return prompt