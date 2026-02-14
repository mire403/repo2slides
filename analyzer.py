from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils import safe_read_text


IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    ".idea",
    ".vscode",
}


@dataclass
class PyObjectInfo:
    name: str
    kind: str  # "class" | "function"
    docstring: str


@dataclass
class FileSummary:
    path: str
    rel_path: str
    language: str
    top_level_docstring: str = ""
    objects: List[PyObjectInfo] = field(default_factory=list)


@dataclass
class RepoSummary:
    root: str
    readme_path: Optional[str]
    main_code_dirs: List[str]
    model_dirs: List[str]
    experiment_dirs: List[str]
    script_dirs: List[str]
    notebook_files: List[str]
    results_files: List[str]
    key_files: List[str]
    project_types: List[str]
    ml_frameworks: List[str]
    language_stats: Dict[str, int]
    entrypoints: List[str]
    file_summaries: List[FileSummary]

    def to_code_structure_text(self) -> str:
        """
        Convert the summary to a compact human-readable text that can be fed to an LLM.
        """
        lines: List[str] = []
        lines.append(f"Root: {self.root}")
        if self.project_types:
            lines.append(f"Detected project types: {', '.join(self.project_types)}")
        if self.key_files:
            lines.append(f"Key files: {', '.join(self.key_files[:20])}")
        if self.language_stats:
            stats = ", ".join(
                f"{k}={v}" for k, v in sorted(self.language_stats.items(), key=lambda x: (-x[1], x[0]))
            )
            lines.append(f"Language stats: {stats}")
        if self.main_code_dirs:
            lines.append(f"Main code dirs: {', '.join(self.main_code_dirs)}")
        if self.model_dirs:
            lines.append(f"Model-related dirs: {', '.join(self.model_dirs)}")
        if self.experiment_dirs:
            lines.append(f"Experiment-related dirs: {', '.join(self.experiment_dirs)}")
        if self.script_dirs:
            lines.append(f"Script dirs: {', '.join(self.script_dirs)}")
        if self.entrypoints:
            lines.append("Inferred entry points:")
            for ep in self.entrypoints[:20]:
                lines.append(f"  - {ep}")
        if self.ml_frameworks:
            lines.append(f"ML frameworks detected: {', '.join(self.ml_frameworks)}")
        if self.notebook_files:
            lines.append("Notebooks:")
            for nb in self.notebook_files[:10]:
                lines.append(f"  - {nb}")
        if self.results_files:
            lines.append("Result/metrics files:")
            for rf in self.results_files[:10]:
                lines.append(f"  - {rf}")

        for fs in self.file_summaries:
            lines.append(f"\n[{fs.rel_path}] ({fs.language})")
            if fs.top_level_docstring:
                lines.append(f"  File docstring: {fs.top_level_docstring[:200].strip()}...")
            for obj in fs.objects[:10]:
                doc_short = (obj.docstring or "").strip().replace("\n", " ")
                if len(doc_short) > 160:
                    doc_short = doc_short[:157] + "..."
                lines.append(f"  {obj.kind} {obj.name}: {doc_short}")
        return "\n".join(lines)


KEY_FILE_NAMES: Dict[str, str] = {
    # Python
    "pyproject.toml": "pyproject.toml",
    "requirements.txt": "requirements.txt",
    "setup.py": "setup.py",
    "setup.cfg": "setup.cfg",
    "poetry.lock": "poetry.lock",
    "pdm.lock": "pdm.lock",
    # Node
    "package.json": "package.json",
    "pnpm-lock.yaml": "pnpm-lock.yaml",
    "yarn.lock": "yarn.lock",
    "package-lock.json": "package-lock.json",
    # Rust
    "cargo.toml": "Cargo.toml",
    # Java / JVM
    "pom.xml": "pom.xml",
    "build.gradle": "build.gradle",
    "build.gradle.kts": "build.gradle.kts",
    # Go
    "go.mod": "go.mod",
    # .NET
    "global.json": "global.json",
}

PROJECT_TYPE_MARKERS: List[Tuple[str, str]] = [
    ("pyproject.toml", "python"),
    ("requirements.txt", "python"),
    ("setup.py", "python"),
    ("package.json", "node"),
    ("cargo.toml", "rust"),
    ("pom.xml", "java"),
    ("build.gradle", "java"),
    ("build.gradle.kts", "java"),
    ("go.mod", "go"),
    ("global.json", "dotnet"),
]

ENTRYPOINT_NAME_HINTS = {
    "main",
    "run",
    "cli",
    "train",
    "fit",
    "evaluate",
    "eval",
    "test",
    "infer",
    "inference",
    "predict",
    "serve",
}


class RepoAnalyzer:
    """
    Analyze a repository structure and extract a light-weight summary:
    - README location
    - main code / models / experiments / scripts directories
    - for Python files: class and function names + docstrings

    It explicitly does NOT inspect the implementation bodies deeply, to keep
    things fast and avoid sending large code to LLMs.
    """

    def __init__(self) -> None:
        pass

    def analyze(self, root: Path) -> RepoSummary:
        root = root.resolve()
        readme_path = self._find_readme(root)

        main_code_dirs: List[str] = []
        model_dirs: List[str] = []
        experiment_dirs: List[str] = []
        script_dirs: List[str] = []
        notebook_files: List[str] = []
        results_files: List[str] = []
        key_files: List[str] = []
        project_types: List[str] = []
        ml_frameworks: set[str] = set()
        language_stats: Dict[str, int] = {}
        entrypoints: List[str] = []
        file_summaries: List[FileSummary] = []

        # Safety caps for large repositories
        max_files_total = 2000
        max_python_files_parsed = 300
        max_file_bytes = 512 * 1024  # 512 KB

        total_files_seen = 0
        python_files_parsed = 0

        # Use os.walk so we can prune ignored directories.
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune ignored dirs in-place.
            dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS and not d.startswith(".")]

            dir_path = Path(dirpath)
            rel_dir = dir_path.relative_to(root)
            parts = {p.lower() for p in rel_dir.parts}
            rel_dir_str = str(rel_dir)

            # Heuristic detection of special dirs
            if rel_dir != Path("."):
                if "models" in parts or "model" in parts:
                    model_dirs.append(rel_dir_str)
                if "experiments" in parts or "experiment" in parts or "exp" in parts:
                    experiment_dirs.append(rel_dir_str)
                if "scripts" in parts or "bin" in parts:
                    script_dirs.append(rel_dir_str)

            # main code dirs: src, project, or package-like directory at repo root
            if rel_dir.parent == Path(".") and rel_dir != Path("."):
                if rel_dir.name in ("src", "app", "project") or self._looks_like_package_dir(dir_path):
                    main_code_dirs.append(rel_dir_str)

            for fn in filenames:
                total_files_seen += 1
                if total_files_seen > max_files_total:
                    break

                path = dir_path / fn
                if not path.is_file():
                    continue

                try:
                    if path.stat().st_size > max_file_bytes:
                        continue
                except OSError:
                    continue

                lang = self._detect_language(path)

                language_stats[lang] = language_stats.get(lang, 0) + 1

                fn_lower = fn.lower()
                if fn_lower in KEY_FILE_NAMES:
                    rel = path.relative_to(root)
                    key_files.append(str(rel))

                rel = path.relative_to(root)
                rel_str = str(rel).replace("\\", "/")

                # Notebook collection
                if lang == "notebook":
                    notebook_files.append(rel_str)

                # Results / metrics file detection (very lightweight heuristics)
                if path.suffix.lower() in {".csv", ".tsv", ".json"}:
                    name_tokens = {fn_lower, rel_str}
                    if any(k in rel_str for k in ("/results/", "/logs/", "/metrics/")) or any(
                        kw in fn_lower for kw in ("result", "metric", "eval", "score", "log")
                    ):
                        results_files.append(rel_str)

                if lang == "python":
                    if python_files_parsed >= max_python_files_parsed:
                        continue
                    fs = self._summarize_python_file(path, root)
                    python_files_parsed += 1
                    if fs:
                        file_summaries.append(fs)
                        entrypoints.extend(self._infer_entrypoints_for_file(fs))
                        # Detect ML frameworks from imports
                        ml_frameworks.update(self._detect_ml_frameworks(path))
                else:
                    # keep only a light breadcrumb for non-python; reduce noise by ignoring common binaries
                    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".tar", ".gz", ".exe"}:
                        continue
                    file_summaries.append(FileSummary(path=str(path), rel_path=str(rel), language=lang))

            if total_files_seen > max_files_total:
                break

        project_types = self._detect_project_types(root, key_files)

        return RepoSummary(
            root=str(root),
            readme_path=str(readme_path) if readme_path else None,
            main_code_dirs=sorted(set(main_code_dirs)),
            model_dirs=sorted(set(model_dirs)),
            experiment_dirs=sorted(set(experiment_dirs)),
            script_dirs=sorted(set(script_dirs)),
            notebook_files=sorted(set(notebook_files)),
            results_files=sorted(set(results_files)),
            key_files=sorted(set(key_files)),
            project_types=project_types,
            ml_frameworks=sorted(ml_frameworks),
            language_stats=language_stats,
            entrypoints=sorted(set(entrypoints)),
            file_summaries=file_summaries,
        )

    @staticmethod
    def _find_readme(root: Path) -> Optional[Path]:
        for name in ("README.md", "README.MD", "Readme.md", "README", "readme.md"):
            candidate = root / name
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _detect_language(path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".py":
            return "python"
        if suffix in {".ipynb"}:
            return "notebook"
        if suffix in {".md", ".rst"}:
            return "markdown"
        if suffix in {".json", ".yml", ".yaml"}:
            return "config"
        if suffix in {".cpp", ".cc", ".cxx", ".h", ".hpp"}:
            return "cpp"
        if suffix in {".js", ".jsx"}:
            return "javascript"
        if suffix in {".ts", ".tsx"}:
            return "typescript"
        if suffix in {".java"}:
            return "java"
        return "other"

    @staticmethod
    def _detect_project_types(root: Path, key_files: List[str]) -> List[str]:
        """
        Detect high-level project type(s) based on key file markers.
        """
        types: set[str] = set()

        key_lower = {Path(p).name.lower() for p in key_files}
        for marker, ptype in PROJECT_TYPE_MARKERS:
            if marker.lower() in key_lower:
                types.add(ptype)

        # heuristic tag: presence of notebooks directory hints "research" style repo
        if (root / "notebooks").exists() or (root / "notebook").exists():
            types.add("research")

        return sorted(types)

    @staticmethod
    def _looks_like_package_dir(path: Path) -> bool:
        """
        A very small heuristic to guess "is this directory the root of a Python package?".
        """
        if not path.is_dir():
            return False
        if (path / "__init__.py").exists():
            return True
        # e.g. src/package_name
        for child in path.iterdir():
            if child.is_dir() and (child / "__init__.py").exists():
                return True
        return False

    def _summarize_python_file(self, path: Path, root: Path) -> Optional[FileSummary]:
        src = safe_read_text(path)
        if not src.strip():
            return None
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return None

        rel = path.relative_to(root)

        top_doc = ast.get_docstring(tree) or ""
        objects: List[PyObjectInfo] = []

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node) or ""
                objects.append(PyObjectInfo(name=node.name, kind="class", docstring=doc))
            elif isinstance(node, ast.FunctionDef):
                doc = ast.get_docstring(node) or ""
                objects.append(PyObjectInfo(name=node.name, kind="function", docstring=doc))

        return FileSummary(
            path=str(path),
            rel_path=str(rel),
            language="python",
            top_level_docstring=top_doc,
            objects=objects,
        )

    def _infer_entrypoints_for_file(self, fs: FileSummary) -> List[str]:
        """
        Infer entry point signals from file path and object names.
        """
        rel_lower = fs.rel_path.replace("\\", "/").lower()
        fname = Path(fs.rel_path).name.lower()
        hits: List[str] = []

        # File name hints
        if any(key in fname for key in ENTRYPOINT_NAME_HINTS):
            hits.append(f"{fs.rel_path} (file)")

        # Object name hints
        for obj in fs.objects:
            n = obj.name.lower()
            if n in ENTRYPOINT_NAME_HINTS or any(
                n.startswith(prefix) for prefix in ("train_", "eval_", "run_", "infer_", "predict_")
            ):
                hits.append(f"{fs.rel_path}: {obj.kind} {obj.name}()")

        # Directory hints for experiments/scripts
        if any(seg in rel_lower for seg in ("/experiments/", "/experiment/", "/scripts/", "/benchmarks/", "/results/")):
            hits.append(f"{fs.rel_path} (experiment/script)")

        return hits

    @staticmethod
    def _detect_ml_frameworks(path: Path) -> set[str]:
        """
        Detect ML frameworks from import statements in Python files.
        """
        frameworks: set[str] = set()
        src = safe_read_text(path)
        if not src.strip():
            return frameworks

        src_lower = src.lower()
        # PyTorch
        if any(imp in src_lower for imp in ("import torch", "from torch", "import pytorch")):
            frameworks.add("PyTorch")
        # TensorFlow
        if any(imp in src_lower for imp in ("import tensorflow", "from tensorflow", "import tf", "from tf")):
            frameworks.add("TensorFlow")
        # Keras (standalone or TF)
        if any(imp in src_lower for imp in ("import keras", "from keras")):
            frameworks.add("Keras")
        # Scikit-learn
        if any(imp in src_lower for imp in ("import sklearn", "from sklearn", "import sklearn")):
            frameworks.add("Scikit-learn")
        # XGBoost
        if any(imp in src_lower for imp in ("import xgboost", "from xgboost")):
            frameworks.add("XGBoost")
        # LightGBM
        if any(imp in src_lower for imp in ("import lightgbm", "from lightgbm")):
            frameworks.add("LightGBM")
        # JAX
        if any(imp in src_lower for imp in ("import jax", "from jax")):
            frameworks.add("JAX")
        # HuggingFace Transformers
        if any(imp in src_lower for imp in ("import transformers", "from transformers", "from huggingface")):
            frameworks.add("HuggingFace")

        return frameworks

