import json
def _parse_json_response(response: str, default=None):
    """Parse JSON from LLM response, handling markdown code blocks."""
    if default is None:
        default = []
    try:
        # Try direct parse
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code block
    if "```" in response:
        try:
            start = response.find("```")
            end = response.rfind("```")
            if start != end:
                code = response[start:end]
                # Remove or ``` prefix
                if code.startswith(""):
                    code = code[7:]
                elif code.startswith("```"):
                    code = code[3:]
                return json.loads(code.strip())
        except json.JSONDecodeError:
            pass
    
    # Try finding JSON array or object
    for start_char, end_char in [('[', ']'), ('{', '}')]:
        start = response.find(start_char)
        end = response.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(response[start:end+1])
            except json.JSONDecodeError:
                continue
    
    return default