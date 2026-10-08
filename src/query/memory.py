"""
src/query/memory.py — Clarify-once persistent location memory.

Core hackathon requirement:
- If user asks about "main gate" and it's unknown → ask user once
- Store the mapping permanently in SQLite
- Never ask again after learning

This module handles the lookup and learning side.
The UI handles the asking side.
"""
from __future__ import annotations

import logging
from typing import Optional

from src import database as db

logger = logging.getLogger(__name__)


def resolve_location_to_camera(location_term: str) -> Optional[str]:
    """
    Attempt to resolve a location term to a camera_id.

    Returns camera_id if known, None if unknown (needs clarification).
    """
    if not location_term:
        return None

    # Check direct aliases and knowledge table
    entry = db.resolve_location(location_term)
    if entry:
        camera_id = entry.get("camera_id")
        logger.info("Location '%s' resolved to camera %s", location_term, camera_id)
        return camera_id

    return None


def learn_location(term: str, camera_id: str,
                   aliases: Optional[list] = None,
                   region_json: Optional[str] = None) -> None:
    """
    Permanently store a location → camera mapping.

    Args:
        term: The location name (e.g., "main gate")
        camera_id: The camera it maps to
        aliases: Alternative names for this location
        region_json: Optional JSON polygon/rectangle within the camera
    """
    term = term.lower().strip()
    db.save_knowledge(term, camera_id, region_json=region_json)
    logger.info("Learned: '%s' → %s", term, camera_id)

    if aliases:
        for alias in aliases:
            alias = alias.lower().strip()
            if alias and alias != term:
                db.add_alias(term, alias)
                logger.info("  Alias: '%s' → '%s'", alias, term)


def get_all_learned_locations() -> list:
    """Get all learned location mappings."""
    return db.get_all_knowledge()


def forget_location(term: str) -> None:
    """Remove a learned location mapping."""
    db.delete_knowledge(term)
    logger.info("Forgot location: '%s'", term)


def is_location_known(term: str) -> bool:
    """Check if a location term is known without resolving it."""
    return resolve_location_to_camera(term) is not None
