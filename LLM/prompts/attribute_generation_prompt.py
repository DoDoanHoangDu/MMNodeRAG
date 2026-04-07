def attribute_generation_prompt(entities, semantic_units, relationships):
    return f"""
You are given a set of synonymous entities (different names referring to the same underlying entity), along with related semantic units and relationships.

Generate a concise, factual summary using only the provided information.

The summary must:
- Treat all listed entities as the same entity and consolidate them into a single coherent description.
- Establish a primary name (use the most descriptive or frequent one) and avoid repeating all variants unnecessarily.
- Extract and consolidate the most important attributes from the semantic units.
- Include the most significant relationships.
- Eliminate redundancy and merge overlapping information only when they clearly refer to the same attribute.
- Prioritize high-signal, defining information over minor details.
- Do not infer or introduce information not explicitly supported by the inputs.
- Prioritize factual accuracy, clarity, and information density.

Output requirements:
- A single paragraph
- Neutral, encyclopedic tone (similar to a Wikipedia article)
- No labels, metadata, or commentary
- Output length must not exceed 500 words
- If the input is limited or sparse, produce a shorter summary without inventing information.

--Entities--
{entities}

--Related Semantic Units--
{semantic_units}

--Related Relationships--
{relationships}
"""