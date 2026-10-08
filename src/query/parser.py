"""
src/query/parser.py — Natural language query parser.

Priority order:
1. Gemini API (if GEMINI_API_KEY set)
2. OpenAI API (if OPENAI_API_KEY set)
3. Rule-based parser (always available, no API needed)

The system MUST work without any API key.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from src.config import cfg
from src.query.schema import StructuredQuery

logger = logging.getLogger(__name__)

# COCO classes the system supports
KNOWN_CLASSES = {
    "person", "car", "bicycle", "motorcycle", "bus", "truck",
    "backpack", "handbag", "suitcase",
    # Aliases
    "vehicle", "automobile", "auto", "van", "bike", "motorbike",
    "human", "people", "man", "woman", "child", "kid", "pedestrian",
    "bag", "purse", "luggage",
}

KNOWN_COLORS = {
    "red", "blue", "green", "white", "black", "gray", "grey",
    "yellow", "orange", "brown", "purple", "pink", "dark", "light",
}

KNOWN_ATTRIBUTES = KNOWN_COLORS | {
    "small", "large", "tall", "short", "wearing", "carrying",
    "backpack", "hat", "jacket", "shirt", "glasses",
}

# Time expression patterns
TIME_PATTERNS = [
    r"in the last \d+ (hour|minute|second|hr|min|sec)s?",
    r"last \d+ (hour|minute|second|hr|min|sec)s?",
    r"last (hour|minute|second|day)",
    r"(today|yesterday|this morning|this afternoon|this evening)",
    r"between \d{1,2}:\d{2} and \d{1,2}:\d{2}",
    r"at \d{1,2}:\d{2}",
    r"around \d{1,2}:\d{2}",
    r"first \d+ (hour|minute|second|hr|min|sec)s?",
]


def _extract_time_range(text: str) -> Optional[str]:
    """Extract time expression from query text."""
    text_lower = text.lower()
    for pattern in TIME_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            return m.group(0)
    return None


def _extract_cameras(text: str) -> list[str]:
    """Extract explicit camera references."""
    cameras = []
    # Pattern: "camera 1", "cam 2", "CAM_01"
    for m in re.finditer(r'\b(?:camera|cam)[_\s]?(\d+)\b', text, re.IGNORECASE):
        num = int(m.group(1))
        cameras.append(f"CAM_{num:02d}")
    # Direct CAM_XX references
    for m in re.finditer(r'\bCAM[_-]?(\d+)\b', text, re.IGNORECASE):
        cam_id = f"CAM_{int(m.group(1)):02d}"
        if cam_id not in cameras:
            cameras.append(cam_id)
    return cameras


def _rule_based_parse(query: str) -> StructuredQuery:
    """
    Deterministic rule-based query parser.
    Works without any API. Always produces a valid result.
    """
    text = query.lower()
    tokens = re.findall(r'\b\w+\b', text)
    token_set = set(tokens)

    object_class = None
    attributes = []

    # Check for person pronouns first
    for token in tokens:
        if token in ("someone", "somebody", "anyone", "anybody"):
            object_class = "person"
            break

    # "carrying" / "wearing" hints
    carrying_m = re.search(r"carry(?:ing)?\s+(?:a\s+)?(\w+)", text)
    if carrying_m:
        item = carrying_m.group(1)
        if item in KNOWN_CLASSES or item in ("backpack", "bag", "handbag", "suitcase", "box", "package"):
            object_class = "person"
            if item not in attributes:
                attributes.append(item)

    # Wearing hints
    wearing_m = re.search(r"wear(?:ing)?\s+(?:a\s+)?(\w+(?:\s+\w+)?)\s+(?:shirt|jacket|hat|dress|coat|top)", text)
    if wearing_m:
        color = wearing_m.group(1).strip()
        if color in KNOWN_COLORS and color not in attributes:
            attributes.append(color)
        object_class = "person"

    # Extract object class if not set
    if object_class is None:
        for token in tokens:
            if token in KNOWN_CLASSES:
                object_class = token
                break

    # Extract color attributes
    for token in tokens:
        if token in KNOWN_COLORS:
            if token not in attributes:
                attributes.append(token)

    # Extract other visual attributes
    visual_attrs = ["backpack", "handbag", "suitcase", "hat", "jacket", "shirt"]
    for attr in visual_attrs:
        if attr in text and attr not in attributes:
            if attr in KNOWN_CLASSES and object_class is None:
                object_class = attr
            elif attr not in attributes:
                attributes.append(attr)

    # Extract location
    location = None
    location_patterns = [
        r"(?:at|near|by|through|in|around)\s+(?:the\s+)?([a-z][a-z\s]+?)(?:\s+in|\s+at|\s+on|\s*[.,?]|$)",
        r"\b(?:main\s+gate|parking(?:\s+lot|\s+area)?|entrance|exit|lobby|reception|corridor|hallway|stairway|back\s+entrance|front\s+gate)\b",
    ]
    for pattern in location_patterns:
        m = re.search(pattern, text)
        if m:
            candidate = m.group(0) if m.lastindex is None else m.group(1)
            candidate = candidate.strip().rstrip(".,?").strip()
            non_locations = {
                "the", "a", "an", "it", "that", "this", "there", "here", "area", "place", "building",
                "camera", "cameras", "any camera", "any cameras", "all cameras", "every camera",
                "any of the cameras", "all of the cameras", "one of the cameras", "each camera",
                "video", "videos", "footage", "screen", "feed", "feeds", "the video", "the footage",
                "the screen", "the feed", "the cameras", "the camera", "recorded footage"
            }
            if candidate and candidate not in non_locations and len(candidate) > 2:
                if not re.search(r'\b(?:cameras?|footage|videos?|feeds?)\b', candidate):
                    location = candidate
                    break

    # Named locations
    named_locs = ["main gate", "back entrance", "side entrance", "parking lot",
                  "parking area", "front entrance", "main entrance", "lobby",
                  "reception", "corridor"]
    for loc in named_locs:
        if loc in text:
            location = loc
            break

    # Extract cameras
    cameras = _extract_cameras(query)

    # Extract time
    time_range = _extract_time_range(query)

    # Intent
    intent = "find_event"
    if any(w in token_set for w in ("count", "how many", "number")):
        intent = "count"
    elif any(w in token_set for w in ("where", "location", "track", "follow")):
        intent = "find_location"
    elif any(w in token_set for w in ("when", "time")):
        intent = "find_time"

    return StructuredQuery(
        object_class=object_class,
        attributes=attributes,
        location=location,
        cameras=cameras,
        time_range=time_range or "all",
        intent=intent,
        raw_query=query,
    )


_GEMINI_COOLDOWN_UNTIL: float = 0.0


def _gemini_parse(query: str) -> Optional[StructuredQuery]:
    """Parse using Gemini API (REST endpoint, no heavy grpc dependency)."""
    global _GEMINI_COOLDOWN_UNTIL
    import time
    if not cfg.gemini_api_key or time.time() < _GEMINI_COOLDOWN_UNTIL:
        return None
    try:
        import requests

        prompt = f"""You are a video surveillance query parser.
Parse this query into a JSON object with these exact fields:
- object_class: the main object (person, car, bicycle, motorcycle, bus, truck, backpack, handbag, suitcase, or null)
- attributes: list of visual attributes (colors, items like backpack, etc.)
- location: named location or null (e.g., "main gate", "parking", etc.)
- cameras: list of camera IDs explicitly mentioned (e.g., ["CAM_01"])
- time_range: time expression or "all" (e.g., "last 1 hour", "all")
- intent: one of find_event, find_location, find_time, count

Query: "{query}"

Return ONLY valid JSON. No markdown fences, no explanation."""

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={cfg.gemini_api_key}"
        resp = requests.post(
            url,
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=5,
        )
        if resp.status_code != 200:
            if resp.status_code == 429:
                _GEMINI_COOLDOWN_UNTIL = time.time() + 300
                logger.info("Gemini quota reached (429), switching to rule-based parser")
            else:
                logger.warning("Gemini API error %d: %s", resp.status_code, resp.text[:200])
            return None

        result_json = resp.json()
        candidates = result_json.get("candidates", [])
        if not candidates:
            return None

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return None

        text = parts[0].get("text", "").strip()
        text = re.sub(r"```(?:json)?\n?", "", text).strip("`").strip()

        data = json.loads(text)
        return StructuredQuery(raw_query=query, **data)

    except Exception as e:
        logger.warning("Gemini parse failed (%s), falling back to rule-based", e)
        return None


def _openai_parse(query: str) -> Optional[StructuredQuery]:
    """Parse using OpenAI API."""
    if not cfg.openai_api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=cfg.openai_api_key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": """Parse surveillance queries into JSON with:
object_class, attributes (list), location, cameras (list), time_range, intent"""},
                {"role": "user", "content": f'Query: "{query}"'},
            ],
            max_tokens=200,
        )
        data = json.loads(response.choices[0].message.content)
        return StructuredQuery(raw_query=query, **data)

    except Exception as e:
        logger.warning("OpenAI parse failed (%s), falling back to rule-based", e)
        return None


def _groq_parse(query: str) -> Optional[StructuredQuery]:
    """Parse using Groq API (super fast, high rate limits)."""
    if not cfg.groq_api_key:
        return None
    try:
        import requests

        prompt = f"""You are an expert CCTV surveillance natural language query parser.
Parse the following surveillance question into a single valid JSON object.
Fields:
- object_class: main object detected (one of: "person", "car", "bicycle", "motorcycle", "bus", "truck", "backpack", "handbag", "suitcase", or null)
- attributes: list of visual attributes like colors ("red", "white", "black", etc.) or items ("backpack")
- location: specific physical named location like "main gate", "parking area", "front entrance", or null. Note: phrases like "any of the cameras", "any camera", "all cameras", "the cameras", "video", "footage" mean NO location constraint, so return null.
- cameras: explicit camera IDs if mentioned like ["CAM_01"], otherwise []
- time_range: time phrase like "last hour", "last 10 minutes", or "all"
- intent: "find_event", "find_location", "find_time", or "count"

Query: "{query}"

Return ONLY valid JSON. No markdown fences, no explanation."""

        url = "https://api.groq.com/openai/v1/chat/completions"
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {cfg.groq_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": cfg.groq_model,
                "messages": [
                    {"role": "system", "content": "You are a CCTV query parser. Respond ONLY with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 200,
                "response_format": {"type": "json_object"},
            },
            timeout=8,
        )
        if resp.status_code != 200:
            logger.warning("Groq API error %d: %s", resp.status_code, resp.text[:200])
            return None

        result_json = resp.json()
        choices = result_json.get("choices", [])
        if not choices:
            return None

        content = choices[0].get("message", {}).get("content", "").strip()
        content = re.sub(r"```(?:json)?\n?", "", content).strip("`").strip()

        data = json.loads(content)
        return StructuredQuery(raw_query=query, **data)

    except Exception as e:
        logger.warning("Groq parse failed (%s), falling back", e)
        return None


class QueryParser:
    """
    Multi-provider query parser with graceful fallback.

    Priority: Groq → Gemini → OpenAI → Rule-based
    """

    def parse(self, query: str) -> StructuredQuery:
        """
        Parse a natural language query.
        Always returns a valid StructuredQuery.
        """
        if not query or not query.strip():
            return StructuredQuery(raw_query=query, intent="find_event")

        # 1. Try Groq API first (ultra-fast, high limits)
        if cfg.groq_api_key:
            result = _groq_parse(query)
            if result:
                return result

        # 2. Try Gemini API
        if cfg.gemini_api_key:
            result = _gemini_parse(query)
            if result:
                return result

        # 3. Try OpenAI API
        if cfg.openai_api_key:
            result = _openai_parse(query)
            if result:
                return result

        # 4. Always available rule-based fallback
        return _rule_based_parse(query)
