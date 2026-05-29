from rapidfuzz import fuzz
from unidecode import unidecode
import re

STOPWORDS = {"and", "or", "the", "of"}

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.casefold()
    text = unidecode(text)
    text = re.sub(r'-', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text_tokens = [word for word in text.split() if word not in STOPWORDS]
    text = ' '.join(text_tokens)
    return text

def score_string(answer, acceptable_answer):
    if not isinstance(answer, str) or not isinstance(acceptable_answer, str):
        return 0
    if answer == acceptable_answer or answer in acceptable_answer or acceptable_answer in answer:
        return 100
    normalized_answer = normalize_text(answer)
    normalized_acceptable = normalize_text(acceptable_answer)
    if normalized_answer == normalized_acceptable or normalized_acceptable in normalized_answer or normalized_answer in normalized_acceptable:
        return 100
    partial_ratio = fuzz.partial_ratio(normalized_answer, normalized_acceptable)
    set_ratio = fuzz.token_set_ratio(normalized_answer, normalized_acceptable)
    return max(partial_ratio, set_ratio)

def score_numeric(answer, acceptable_answer):
    if not isinstance(answer, (int, float, str)) or not isinstance(acceptable_answer, dict):
        return 0
    numbers = re.findall(r'-?\d+(?:\.\d+)?', answer)
    if len(numbers) == 0:
        return 0
    value = acceptable_answer.get("value")
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return 0
    range_value = acceptable_answer.get("range")
    if not isinstance(value, (int, float)) or not (isinstance(range_value, list) and 2 >= len(range_value) >= 1) or not all(isinstance(x, (int, float)) for x in range_value):
        return 0
    for num in numbers:
        num = float(num)
        if num == value:
            return 100
        if not range_value[0] <= num <= range_value[-1]:
            return 0
    return 100

def fuzzy_accuracy(answer, acceptable_answers, threshold=95):
    for acceptable_answer in acceptable_answers:
        score = score_string(answer, acceptable_answer) if isinstance(acceptable_answer, str) else score_numeric(answer, acceptable_answer)
        if score >= threshold:
            return 1
    return 0