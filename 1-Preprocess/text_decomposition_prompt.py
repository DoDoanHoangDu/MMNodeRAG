def text_decomposition_prompt(text):
    base = f"""--Goal--
Given a text, segment it into multiple semantic units, each containing detailed descriptions of specific events or activities.
Perform the following tasks:
--Steps--
1. Provide a summary for each semantic unit while retaining all crucial details relevant to the original context.
2. Extract all entities directly from the original text of each semantic unit, not from the paraphrased. Format each entity name in UPPERCASE. You should extract all entities including times, locations, people, organizations and all kinds of entities.
3. From the entities extracted in Step 2, list all relationships within the semantic unit and the corresponding original context in the form of string separated by comma: "ENTITY_A, RELATION_TYPE, ENTITY_B". The RELATION_TYPE could be a descriptive sentence, while the entities involved in the relationship must come from the entity names extracted in Step 2. Please make sure the string contains three elements representing two entities and the relationship type.
--Requirement--
1. Temporal Entities: Represent time entities based on the available details without filling in missing parts. Use specific formats based on what parts of the date or time are mentioned in the text.
2. Each semantic unit should be represented as a dictionary containing three keys: semantic_unit (a paraphrased summary of each semantic unit), entities (a list of entities extracted directly from the original text of each semantic unit, formatted in UPPERCASE), and relationships (a list of extracted relationship strings that contain three elements, where the relationship type is a descriptive sentence). All these dictionaries should be stored in a list to facilitate management and access.
3. Coreference Resolution:
- Resolve all pronouns and referring expressions (e.g., he, she, it, they, this, that, the company, the organization) to their explicit entity names.
- Do NOT include pronouns or vague references as entities.
- If a semantic unit contains references whose antecedents appear in earlier semantic units, include those antecedent entities explicitly in the current unit’s entity list.
- If a sentence refers to an entity mentioned earlier in the text or in a previous semantic unit, use the canonical entity name.
- All entities used in relationships must be explicit, fully resolved, and appear in the entities list.

--Example--
Text:
In September 2024, Dr. Emily Roberts traveled to Paris to attend the International Conference on Renewable Energy. During her visit, she explored partnerships with several European companies and presented her latest research on solar panel efficiency improvements. Meanwhile, on the other side of the world, her colleague, Dr. John Miller, was conducting fieldwork in the Amazon Rainforest. He documented several new species and observed the effects of deforestation on the local wildlife. Both scholars' work is essential in their respective fields and contributes significantly to environmental conservation efforts.

Output:
[
{{
semantic_unit: In September 2024, Dr. Emily Roberts attended the International Conference on Renewable Energy in Paris, where she presented her research on solar panel efficiency improvements and explored partnerships with European companies.,
entities: ["DR. EMILY ROBERTS", "2024-09", "PARIS", "INTERNATIONAL CONFERENCE ON RENEWABLE ENERGY", "EUROPEAN COMPANIES", "SOLAR PANEL EFFICIENCY"], 
relationships:
[
"DR. EMILY ROBERTS, attended, INTERNATIONAL CONFERENCE ON RENEWABLE ENERGY",
"DR. EMILY ROBERTS, explored partnerships with, EUROPEAN COMPANIES",
"DR. EMILY ROBERTS, presented research on, SOLAR PANEL EFFICIENCY"
]
}},
{{
semantic_unit: Dr. John Miller conducted fieldwork in the Amazon Rainforest, documenting several new species and observing the effects of deforestation on local wildlife.",
entities: ["DR. JOHN MILLER", "AMAZON RAINFOREST", "NEW SPECIES", "DEFORESTATION", "LOCAL WILDLIFE"],
relationships:
[
"DR. JOHN MILLER, conducted fieldwork in, AMAZON RAINFOREST",
"DR. JOHN MILLER, documented, NEW SPECIES",
"DR. JOHN MILLER, observed the effects of, DEFORESTATION on LOCAL WILDLIFE"
]
}},
{{
semantic_unit: "The work of both Dr. Emily Roberts and Dr. John Miller is crucial in their respective fields and contributes significantly to environmental conservation efforts.",
entities: ["DR. EMILY ROBERTS", "DR. JOHN MILLER", "ENVIRONMENTAL CONSERVATION"], 
relationships: 
[
"DR. EMILY ROBERTS, contributes to, ENVIRONMENTAL CONSERVATION",
"DR. JOHN MILLER, contributes to, ENVIRONMENTAL CONSERVATION"
]
}}
]
--Real Input--
Text:{text}
"""
    return base

