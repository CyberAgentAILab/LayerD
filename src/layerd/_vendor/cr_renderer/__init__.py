"""
cr-renderer - Vendored from https://github.com/CyberAgentAILab/cr-renderer
Copyright (c) CyberAgent, Inc.
Licensed under Apache License 2.0

This is a bundled dependency for LayerD. Do not import from layerd._vendor directly.
"""
from .fonts import FontManager
from .renderer import CrelloV4Renderer, CrelloV5Renderer

__all__ = ["CrelloV5Renderer", "CrelloV4Renderer", "FontManager"]
