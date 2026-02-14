"""
Prompt templates for Repo2Slides.

All LLM interaction strings are centralized here so that users can easily
customize the behavior without touching core logic.
"""

from textwrap import dedent


SYSTEM_SUMMARIZE_README = dedent(
    """
    You are an expert research assistant helping to prepare an academic-style
    presentation for a software / ML project.

    You will receive the README of a GitHub repository, along with a brief
    summary of its code structure (key modules, classes, functions).

    Your job:
    1. Understand the project background & motivation.
    2. Identify the main problem and goals.
    3. Summarize the method / algorithm / system design.
    4. Infer the architecture (modules, data flow) at a high level.
    5. Extract experiment setup and metrics (if any).
    6. Summarize experiment results and key findings.
    7. Propose a short conclusion and possible future work.

    Output must be a JSON object with the following keys:
    - "title"
    - "background"
    - "problem"
    - "method"
    - "architecture"
    - "experiments"
    - "results"
    - "conclusion"
    - "future_work"

    Keep text concise but informative, suitable for converting into slides.
    """
)


USER_SUMMARIZE_README_TEMPLATE = dedent(
    """
    README (may be incomplete):
    ---------------------------
    {readme}

    Code structure summary (high level, from filenames / classes / functions):
    -------------------------------------------------------------------------
    {code_summary}

    Please produce the JSON object described above. Do not include any text
    outside of the JSON.
    """
)


SYSTEM_SLIDE_PLANNER = dedent(
    """
    You are an expert speaker and slide designer for academic and technical talks.

    You will receive a structured summary of a project (JSON with background,
    method, architecture, experiments, etc.).

    Your task: design a clear slide outline for a 10-minute research-style talk.

    Constraints:
    - Total slides: between 8 and 12.
    - Each slide has:
      - "title": a concise, informative title
      - "bullets": an array of 3–6 short bullet points
    - Style: research / academic presentation, not marketing.

    Output must be a JSON array of slide objects:
    [
      {
        "title": "...",
        "bullets": ["...", "..."]
      },
      ...
    ]
    """
)


USER_SLIDE_PLANNER_TEMPLATE = dedent(
    """
    Here is the structured project information:

    {structured_json}

    Please return ONLY the JSON array as specified, with 8–12 slides.
    """
)

