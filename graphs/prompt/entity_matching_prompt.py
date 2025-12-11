def entity_matching_prompt(entity, relevant_entities):
    prompt = f"""You are given:
- an ORIGINAL entity string (single value)
- a LIST_OF_ENTITIES (a list of strings)

Task:
Return a list containing only the items from LIST_OF_ENTITIES that are IMMEDIATELY RELATED to the ORIGINAL entity. Output must be exactly one line containing only the list (no explanation, no punctuation outside the list, no surrounding text).

Definition — IMMEDIATELY RELATED:
Consider an entity from LIST_OF_ENTITIES immediately related to ORIGINAL if any of these deterministic rules apply (apply them in order; stop when one rule clearly matches):

1. Exact match (case-insensitive) after trimming whitespace.
2. Same after removing common legal/company suffixes and punctuation (remove tokens: "inc", "inc.", "llc", "ltd", "co", "corp", "corporation", "company", "plc") and comparing case-insensitively.
3. Acronym/initials match: if the ORIGINAL's initials (first letters of each word) equal the candidate (ignoring dots and case), or vice versa. Example: "World Health Organization" and "WHO".
4. Substring containment (case-insensitive) where one string contains the other as a whole-word sequence after removing punctuation. Example: "International Business Machines" and "IBM Research" (match only if containment shows direct reference).
5. High lexical variation: the candidate differs only by small common orthographic variations (character insertions/deletions/replacements) such that it appears to be a formatting variant or abbreviated form (e.g., "U.S.A." and "USA", "O'Neil" and "Oneil"). Use this rule conservatively.
6. Explicit alias tokens: candidate contains tokens like "aka", "also known as", "/", "—" that directly link names (e.g., "Meta (formerly Facebook)").

Rules for selection & output:
- Only choose items from the provided LIST_OF_ENTITIES.
- Return the matched items in the same order they appear in LIST_OF_ENTITIES.
- Output entity strings exactly as they appear in LIST_OF_ENTITIES (do not canonicalize or change punctuation).
- If no items match, output an empty list: `[]`.
- The output must be a single line with a valid list (double quotes for strings recommended but single quotes are acceptable).
- Do not include duplicates; if the same string appears multiple times in LIST_OF_ENTITIES keep its first occurrence only.
- Do not use external knowledge; decide based only on the ORIGINAL and LIST_OF_ENTITIES and the rules above.

Input format (the model will receive these two variables):
ORIGINAL = "{entity}"
LIST_OF_ENTITIES = {relevant_entities}

Output example 1 (synonym/alias):
ORIGINAL = "IBM"
LIST_OF_ENTITIES = ["International Business Machines", "I.B.M.", "IBMer", "Microsoft"]
Output:
["International Business Machines", "I.B.M."]

Output example 2 (acronym and containment):
ORIGINAL = "World Health Organization"
LIST_OF_ENTITIES = ["WHO", "World Health Org.", "Health Department", "World Health"]
Output:
["WHO", "World Health Org.", "World Health"]

Output example 3 (no match):
ORIGINAL = "Acme Corp"
LIST_OF_ENTITIES = ["Beta LLC", "Gamma Inc"]
Output:
[]
"""
    return prompt