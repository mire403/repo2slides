"""
Repo2Slides - Turn a GitHub repo into slides.

This package provides the core building blocks:

- RepoAnalyzer: analyze repository structure
- ContentExtractor: extract structured description from README & code
- SlidePlanner: plan slide pages (8–12 slides)
- SlideGenerator: generate markdown / pptx slides
"""

from .analyzer import RepoAnalyzer
from .extractor import ContentExtractor
from .planner import SlidePlanner
from .generator import SlideGenerator

__all__ = [
    "RepoAnalyzer",
    "ContentExtractor",
    "SlidePlanner",
    "SlideGenerator",
]

