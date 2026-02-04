import json
import re
from typing import Optional, Dict

def extract_json(text: str) -> Optional[Dict]:
    """Extracts a JSON object from a string that is formatted as a code block."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
    return None