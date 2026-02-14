from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .analyzer import RepoSummary
from .prompts import (
    SYSTEM_SUMMARIZE_README,
    USER_SUMMARIZE_README_TEMPLATE,
)
from .utils import LLMClient, dict_get_first, safe_read_text, shorten


DEFAULT_STRUCTURED_TEMPLATE: Dict[str, str] = {
    "title": "",
    "background": "",
    "problem": "",
    "method": "",
    "architecture": "",
    "experiments": "",
    "results": "",
    "conclusion": "",
    "future_work": "",
}


class ContentExtractor:
    """
    Extract structured project information from README and code summary.

    If an LLM client is provided, it will be used to produce higher quality
    summaries. Otherwise, a simple heuristic fallback will be used.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self.llm_client = llm_client

    def extract(self, repo_summary: RepoSummary) -> Dict[str, str]:
        readme_text = ""
        if repo_summary.readme_path:
            readme_text = safe_read_text(Path(repo_summary.readme_path))
        code_summary_text = repo_summary.to_code_structure_text()

        if self.llm_client:
            return self._extract_with_llm(readme_text, code_summary_text)
        return self._extract_heuristic(readme_text, code_summary_text, repo_summary)

    def _extract_with_llm(self, readme: str, code_summary: str) -> Dict[str, str]:
        user_prompt = USER_SUMMARIZE_README_TEMPLATE.format(
            readme=shorten(readme, 6000),
            code_summary=shorten(code_summary, 4000),
        )
        raw = self.llm_client.chat(system_prompt=SYSTEM_SUMMARIZE_README, user_prompt=user_prompt)

        # be defensive about JSON parsing
        data: Dict[str, Any]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to salvage by finding first {...} block
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(raw[start : end + 1])
                except json.JSONDecodeError:
                    data = {}
            else:
                data = {}

        return self._normalize_structured_dict(data)

    def _extract_heuristic(self, readme: str, code_summary: str, repo: RepoSummary) -> Dict[str, str]:
        """
        Very simple heuristic extraction when no LLM is available.

        Strategy:
        - Use README first heading as title.
        - Use first 2–3 paragraphs as background / problem.
        - Use sections that look like "Method", "Approach" etc. if present.
        - Use code summary as architecture / experiments fallback.
        """
        result: Dict[str, str] = dict(DEFAULT_STRUCTURED_TEMPLATE)

        lines = readme.splitlines()
        title = ""
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                # remove leading #'s
                title = stripped.lstrip("#").strip()
                break
        if not title:
            title = "Project Overview"
        result["title"] = title

        sections = self._split_markdown_sections(lines)

        intro = sections.get("intro", "").strip()
        if not intro:
            intro = "\n".join([ln for ln in lines if ln.strip()][:20]).strip()

        background_text = self._pick_sections(
            sections,
            [
                "background",
                "motivation",
                "introduction",
                "overview",
                "简介",
                "背景",
                "动机",
                "概述",
            ],
            fallback=intro,
        )
        result["background"] = background_text.strip()

        # Multi-language / build system hints
        if repo.project_types:
            result["background"] = (
                result["background"]
                + ("\n\n" if result["background"] else "")
                + f"Tech stack (inferred): {', '.join(repo.project_types)}."
            ).strip()

        # ML framework hints
        if getattr(repo, "ml_frameworks", None) and repo.ml_frameworks:
            framework_note = f"ML frameworks detected: {', '.join(repo.ml_frameworks)}."
            if not result["method"]:
                result["method"] = framework_note
            else:
                result["method"] = (result["method"] + "\n\n" + framework_note).strip()

        problem_text = self._pick_sections(
            sections,
            [
                "problem",
                "task",
                "objective",
                "goals",
                "why",
                "问题",
                "任务",
                "目标",
                "动机",
            ],
            fallback="",
        )
        result["problem"] = problem_text.strip()

        method_text = self._pick_sections(
            sections,
            [
                "method",
                "approach",
                "algorithm",
                "model",
                "pipeline",
                "methodology",
                "方法",
                "算法",
                "模型",
                "流程",
                "原理",
            ],
            fallback="",
        )
        result["method"] = method_text.strip()

        # If method is still empty, infer from entrypoints / key files
        if not result["method"]:
            inferred_method_lines: list[str] = []
            if repo.entrypoints:
                inferred_method_lines.append("Key entry points (inferred):")
                inferred_method_lines.extend([f"- {ep}" for ep in repo.entrypoints[:8]])
            if repo.main_code_dirs:
                inferred_method_lines.append(f"Main implementation dirs: {', '.join(repo.main_code_dirs[:5])}")
            if inferred_method_lines:
                result["method"] = "\n".join(inferred_method_lines).strip()

        arch_text = self._pick_sections(
            sections,
            [
                "architecture",
                "design",
                "system",
                "implementation",
                "framework",
                "structure",
                "架构",
                "系统",
                "设计",
                "实现",
                "框架",
                "结构",
            ],
            fallback="",
        ).strip()
        if code_summary:
            arch_text = (arch_text + "\n\nCode structure:\n" + shorten(code_summary, 3000)).strip()
        result["architecture"] = arch_text

        experiments_text = self._pick_sections(
            sections,
            [
                "experiment",
                "experiments",
                "evaluation",
                "benchmark",
                "training",
                "dataset",
                "setup",
                "实验",
                "评估",
                "训练",
                "数据集",
                "设置",
                "配置",
            ],
            fallback="",
        )
        result["experiments"] = experiments_text.strip()

        # If experiments empty, try infer from dirs + entrypoints + notebooks
        if not result["experiments"]:
            exp_lines: list[str] = []
            if repo.experiment_dirs:
                exp_lines.append(f"Experiment-related dirs (inferred): {', '.join(repo.experiment_dirs[:8])}")
            if repo.script_dirs:
                exp_lines.append(f"Script dirs (inferred): {', '.join(repo.script_dirs[:8])}")
            exp_eps = [ep for ep in repo.entrypoints if "(experiment/script)" in ep.lower() or "train" in ep.lower() or "eval" in ep.lower()]
            if exp_eps:
                exp_lines.append("Potential experiment scripts / functions:")
                exp_lines.extend([f"- {ep}" for ep in exp_eps[:8]])
            if repo.notebook_files:
                exp_lines.append("Jupyter notebooks (often used for experiments / analysis):")
                for nb in repo.notebook_files[:6]:
                    exp_lines.append(f"- {nb}")
            if exp_lines:
                result["experiments"] = "\n".join(exp_lines).strip()

        results_text = self._pick_sections(
            sections,
            [
                "results",
                "ablation",
                "analysis",
                "metrics",
                "performance",
                "结论",
                "结果",
                "消融",
                "分析",
                "指标",
                "性能",
            ],
            fallback="",
        )
        result["results"] = results_text.strip()

        if not result["results"]:
            # Try to summarize results/metrics files if present.
            metrics_summary = self._summarize_results_files(repo)
            if metrics_summary:
                result["results"] = metrics_summary
            elif result["experiments"]:
                result["results"] = (
                    "Results are not explicitly reported in README; please refer to experiment logs / output files."
                )

        conclusion_text = self._pick_sections(
            sections,
            ["conclusion", "summary", "takeaways", "总结", "结论", "小结"],
            fallback="",
        )
        result["conclusion"] = conclusion_text.strip()

        future_text = self._pick_sections(
            sections,
            ["future", "limitations", "roadmap", "todo", "未来", "展望", "局限", "路线图", "待办"],
            fallback="",
        )
        result["future_work"] = future_text.strip()

        # If method/problem are empty, use intro snippet as a fallback seed
        if not result["problem"] and intro:
            result["problem"] = "\n".join(intro.splitlines()[:8]).strip()
        if not result["method"] and code_summary:
            result["method"] = "Key modules and entry points inferred from code structure."

        return result

    @staticmethod
    def _split_markdown_sections(lines: list[str]) -> Dict[str, str]:
        """
        Split README into sections keyed by normalized heading text.
        """
        sections: Dict[str, str] = {}
        current_header = "intro"
        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer, current_header
            content = "\n".join(buffer).strip()
            if content:
                # merge repeated headers
                if current_header in sections:
                    sections[current_header] = (sections[current_header] + "\n\n" + content).strip()
                else:
                    sections[current_header] = content
            buffer = []

        for line in lines:
            s = line.strip()
            if s.startswith("#"):
                flush()
                header_text = s.lstrip("#").strip().lower()
                current_header = header_text if header_text else "intro"
            else:
                # Skip Setext underline noise lines (==== / ----)
                if ContentExtractor._is_setext_underline(s):
                    continue
                buffer.append(line)
        flush()
        return sections

    @staticmethod
    def _pick_sections(sections: Dict[str, str], keywords: list[str], fallback: str = "") -> str:
        """
        Pick and concatenate sections whose header contains any keyword.
        """
        hits: list[str] = []
        for header, text in sections.items():
            h = header.lower()
            if any(k.lower() in h for k in keywords):
                if text.strip():
                    hits.append(text.strip())
        return ("\n\n".join(hits).strip()) or fallback.strip()

    @staticmethod
    def _is_setext_underline(line: str) -> bool:
        s = line.strip()
        if len(s) < 3:
            return False
        return all(ch == "=" for ch in s) or all(ch == "-" for ch in s)

    def _summarize_results_files(self, repo: RepoSummary) -> str:
        """
        Summarize potential results / metrics files (CSV/JSON) with numeric analysis.
        Extracts best/last values for common metrics.
        """
        if not getattr(repo, "results_files", None):
            return ""

        lines: list[str] = []
        root = Path(repo.root)
        max_files = 5

        for rel in repo.results_files[:max_files]:
            path = root / rel
            if not path.exists():
                continue
            suffix = path.suffix.lower()
            if suffix in {".csv", ".tsv"}:
                summary = self._analyze_csv_metrics(path, rel)
                if summary:
                    lines.append(summary)
                else:
                    header = self._read_csv_header(path)
                    if header:
                        cols = ", ".join(header[:8])
                        lines.append(f"{rel}: CSV/TSV with columns [{cols}]")
                    else:
                        lines.append(f"{rel}: CSV/TSV metrics file")
            elif suffix == ".json":
                summary = self._analyze_json_metrics(path, rel)
                if summary:
                    lines.append(summary)
                else:
                    keys = self._read_json_keys(path)
                    if keys:
                        ks = ", ".join(keys[:8])
                        lines.append(f"{rel}: JSON with keys [{ks}]")
                    else:
                        lines.append(f"{rel}: JSON metrics / results file")
            else:
                lines.append(f"{rel}: results/metrics file")

        return "\n".join(lines).strip()

    @staticmethod
    def _read_csv_header(path: Path) -> list[str]:
        import csv

        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        return [c.strip() for c in row if c.strip()]
        except OSError:
            return []
        return []

    @staticmethod
    def _read_json_keys(path: Path) -> list[str]:
        try:
            text = safe_read_text(path)
            if not text.strip():
                return []
            obj = json.loads(text)
        except Exception:
            return []

        if isinstance(obj, dict):
            return [str(k) for k in obj.keys()]
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return [str(k) for k in obj[0].keys()]
        return []

    @staticmethod
    def _analyze_csv_metrics(path: Path, rel: str) -> str:
        """
        Analyze CSV/TSV metrics file: extract best/last values for common metric columns.
        """
        import csv

        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if not rows:
                    return ""

                # Common metric column names (case-insensitive)
                metric_keywords = {
                    "acc": "accuracy",
                    "accuracy": "accuracy",
                    "f1": "f1",
                    "f1_score": "f1",
                    "loss": "loss",
                    "val_loss": "val_loss",
                    "val_acc": "val_accuracy",
                    "val_accuracy": "val_accuracy",
                    "epoch": "epoch",
                    "step": "step",
                }

                # Find metric columns
                headers = [h.lower().strip() for h in reader.fieldnames or []]
                metric_cols: Dict[str, str] = {}
                for h in headers:
                    for kw, label in metric_keywords.items():
                        if kw in h:
                            metric_cols[h] = label
                            break

                if not metric_cols:
                    return ""

                # Extract numeric values and compute best/last
                findings: list[str] = []
                for col, label in list(metric_cols.items())[:5]:  # Limit to 5 metrics
                    values = []
                    for row in rows:
                        val_str = row.get(col, "").strip()
                        try:
                            val = float(val_str)
                            values.append((len(values), val))
                        except (ValueError, TypeError):
                            continue

                    if not values:
                        continue

                    nums = [v[1] for v in values]
                    last_val = nums[-1]
                    if "loss" in label.lower():
                        best_val = min(nums)
                        best_idx = min(range(len(nums)), key=lambda i: nums[i])
                        findings.append(f"{label}: best={best_val:.4f} (epoch {best_idx+1}), last={last_val:.4f}")
                    else:
                        best_val = max(nums)
                        best_idx = max(range(len(nums)), key=lambda i: nums[i])
                        findings.append(f"{label}: best={best_val:.4f} (epoch {best_idx+1}), last={last_val:.4f}")

                if findings:
                    return f"{rel}: " + " | ".join(findings)
                return ""

        except Exception:
            return ""

    @staticmethod
    def _analyze_json_metrics(path: Path, rel: str) -> str:
        """
        Analyze JSON metrics file: extract best/last values for common metric keys.
        """
        try:
            text = safe_read_text(path)
            if not text.strip():
                return ""
            obj = json.loads(text)
        except Exception:
            return ""

        findings: list[str] = []
        metric_keywords = {
            "acc": "accuracy",
            "accuracy": "accuracy",
            "f1": "f1",
            "f1_score": "f1",
            "loss": "loss",
            "val_loss": "val_loss",
            "val_acc": "val_accuracy",
        }

        def extract_metrics(data: Any, prefix: str = "") -> None:
            if isinstance(data, dict):
                for k, v in data.items():
                    k_lower = k.lower()
                    if isinstance(v, (int, float)):
                        for kw, label in metric_keywords.items():
                            if kw in k_lower:
                                findings.append(f"{prefix}{label}={v:.4f}" if prefix else f"{label}={v:.4f}")
                                break
                    elif isinstance(v, (dict, list)):
                        extract_metrics(v, prefix=f"{k}." if prefix else f"{k}.")
            elif isinstance(data, list) and data:
                # If it's a list of metrics (e.g., per-epoch), analyze first and last
                if isinstance(data[0], dict):
                    first = data[0]
                    last = data[-1]
                    for k, v in first.items():
                        k_lower = k.lower()
                        if isinstance(v, (int, float)):
                            for kw, label in metric_keywords.items():
                                if kw in k_lower:
                                    last_val = last.get(k, v)
                                    if isinstance(last_val, (int, float)):
                                        findings.append(f"{label}: first={v:.4f}, last={last_val:.4f}")
                                    break

        extract_metrics(obj)
        if findings:
            return f"{rel}: " + " | ".join(findings[:6])
        return ""

    @staticmethod
    def _normalize_structured_dict(data: Dict[str, Any]) -> Dict[str, str]:
        """
        Ensure that all expected keys exist and are strings.
        """
        result: Dict[str, str] = dict(DEFAULT_STRUCTURED_TEMPLATE)
        for key in DEFAULT_STRUCTURED_TEMPLATE:
            value = data.get(key, "")
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False, indent=2)
            result[key] = str(value).strip()
        return result

