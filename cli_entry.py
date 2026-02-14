from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional

from .analyzer import RepoAnalyzer
from .extractor import ContentExtractor
from .generator import SlideGenerator
from .planner import SlidePlanner
from .utils import LLMConfig, LLMClient, ensure_path, log


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="repo2slides",
        description="Turn a GitHub repo (local clone) into slides.",
    )
    parser.add_argument(
        "repo_path",
        type=str,
        help="Path to the local repository to analyze.",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default="slides.md",
        help="Output file path. Default: slides.md",
    )
    parser.add_argument(
        "--format",
        choices=["md", "pptx"],
        default="md",
        help="Output format: md (default) or pptx (requires python-pptx).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model name. Default: gpt-4o-mini",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key. If not provided, uses OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM usage and use heuristic extraction only (works offline).",
    )
    parser.add_argument(
        "--md-engine",
        choices=["marp", "plain"],
        default="marp",
        help="Markdown slides engine flavor. Default: marp",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="academic",
        help="Markdown/PPT theme name (used in markdown front matter). Default: academic",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "zh"],
        default="en",
        help="Slide language style for heuristic mode (en or zh). Default: en",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages during analysis and generation.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    repo_path = pathlib.Path(args.repo_path).expanduser().resolve()
    if not repo_path.exists() or not repo_path.is_dir():
        print(f"[repo2slides] ERROR: repo_path is not a directory: {repo_path}", file=sys.stderr)
        return 1

    out_path = pathlib.Path(args.out).expanduser().resolve()
    ensure_path(out_path.parent)

    llm_client: Optional[LLMClient] = None
    if not args.no_llm:
        config = LLMConfig(model=args.model, api_key=args.api_key)
        try:
            if args.verbose:
                log(f"Initializing LLM client with model={config.model}")
            llm_client = LLMClient(config)
        except Exception as e:
            print(f"[repo2slides] WARNING: Failed to initialize LLM client: {e}", file=sys.stderr)
            print("[repo2slides] Falling back to heuristic (no-LLM) mode.", file=sys.stderr)
            llm_client = None

    analyzer = RepoAnalyzer()
    if args.verbose:
        log(f"Analyzing repository at: {repo_path}")
    try:
        repo_summary = analyzer.analyze(repo_path)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[repo2slides] ERROR during analysis: {e}", file=sys.stderr)
        return 1

    extractor = ContentExtractor(llm_client=llm_client)
    if args.verbose:
        log("Extracting structured project information (README + code structure)...")
    try:
        structured_content = extractor.extract(repo_summary)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[repo2slides] ERROR during extraction: {e}", file=sys.stderr)
        return 1

    # If LLM is enabled, it's useful to also plan slides with LLM for phrasing.
    planner = SlidePlanner(llm_client=llm_client, language=args.lang)
    if args.verbose:
        log("Planning slide outline...")
    try:
        slide_plan = planner.plan_slides(structured_content)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[repo2slides] ERROR during slide planning: {e}", file=sys.stderr)
        return 1

    generator = SlideGenerator()

    if args.format == "md":
        if args.verbose:
            log("Generating markdown slides...")
        markdown = generator.to_markdown_with_front_matter(
            slide_plan,
            title=structured_content.get("title", "Repo2Slides") or "Repo2Slides",
            author="Repo2Slides (auto-generated)",
            theme=args.theme,
            engine=args.md_engine,
        )
        out_path.write_text(markdown, encoding="utf-8")
        if args.verbose:
            log(f"Markdown slides written to {out_path}")
        else:
            print(f"[repo2slides] Markdown slides written to {out_path}")
        return 0

    try:
        if args.verbose:
            log("Generating PPTX slides...")
        generator.to_pptx(slide_plan, out_path)
        print(f"[repo2slides] PPTX slides written to {out_path}")
        return 0
    except ImportError as e:
        print(f"[repo2slides] ERROR: {e}. Install python-pptx or use --format md.", file=sys.stderr)
        return 1

