"""
DEPRECATED MODULE.

This file is no longer the active CRS pipeline entry point.
It was used during early prototyping.

For the current thesis system and web application, use:

    my_crs.recommender.get_recommendation()

Current architecture:
Dialogue -> KBRD candidate retrieval -> Qwen reranking -> Qwen response generation

This module is intentionally kept only for historical/project-structure clarity.
Do not import it in new code.
"""
