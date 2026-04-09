def high_level_elements_prompt(text):
    prompt = f"""
You will receive a set of text data from the same cluster. Your task is to extract distinct categories of high-level information, such as concepts, themes, relevant theories, potential impacts, and key insights. Each piece of information should include a concise title and a corresponding description, reflecting the unique perspectives within the text cluster.
Please do not attempt to include all possible information; instead, select the elements that have the most significance and diversity in this cluster. Avoid redundant information—if there are highly similar elements, combine them into a single, comprehensive entry. Ensure that the high-level information reflects the varied dimensions within the text, providing a well-rounded overview.
Return the result as multiple well-structured paragraphs, where each paragraph represents one high-level element (with an implicit title and its description integrated naturally). Return only the only the final paragraphs, do not include any commentary or meta-text.
--Clustered text data--
{text}
"""
    return prompt