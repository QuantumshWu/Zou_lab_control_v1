"""XeLaTeX notes/manual generation for the Zou lab front-end."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from importlib import resources
from pathlib import Path
import shutil
import subprocess

from .content.manuals import frontend_manual_body, generate_frontend_manual_figures


@dataclass(frozen=True)
class NotesBuildResult:
    """Paths produced by a notes/manual build."""

    tex_path: Path
    pdf_path: Path
    build_dir: Path
    log_path: Path


def notes_template_dir() -> Path:
    """Return the package directory containing XeLaTeX note templates."""
    return Path(str(resources.files("Zou_lab_control.frontend") / "templates"))


def _copy_template_files(build_dir: Path) -> None:
    template_dir = notes_template_dir()
    for src in template_dir.glob("*"):
        if src.is_file():
            shutil.copy2(src, build_dir / src.name)


def _latex_text(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def write_notes_tex(
    output_dir: str | Path,
    *,
    filename: str = "notes.tex",
    title: str,
    subtitle: str = "",
    description: str = "",
    body: str,
    doc_date: str | None = None,
    template_package: str = "zlc_frontend_notes",
) -> Path:
    """Write a XeLaTeX notes document using the front-end notes style."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_template_files(output_dir)
    tex_path = output_dir / filename
    doc_date = doc_date or date.today().isoformat()
    title_tex = _latex_text(title)
    subtitle_tex = _latex_text(subtitle)
    description_tex = _latex_text(description)
    date_tex = _latex_text(doc_date)
    tex = rf"""\documentclass[11pt,oneside,openany]{{book}}
\usepackage{{{template_package}}}

\begin{{document}}
\frontmatter
\zlctitlepage{{{title_tex}}}{{{subtitle_tex}}}{{{description_tex}}}{{{date_tex}}}
\tableofcontents
\cleardoublepage
\mainmatter

{body}

\end{{document}}
"""
    tex_path.write_text(tex, encoding="utf-8")
    return tex_path


def compile_notes_pdf(
    tex_path: str | Path,
    *,
    runs: int = 2,
    xelatex: str | None = None,
    halt_on_error: bool = True,
) -> NotesBuildResult:
    """Compile a notes ``.tex`` file with XeLaTeX."""
    tex_path = Path(tex_path).resolve()
    build_dir = tex_path.parent
    exe = xelatex or shutil.which("xelatex")
    if exe is None:
        raise RuntimeError("xelatex was not found on PATH.")

    log_path = build_dir / f"{tex_path.stem}.build.log"
    combined = []
    for _ in range(max(1, int(runs))):
        cmd = [exe, "-interaction=nonstopmode"]
        if halt_on_error:
            cmd.append("-halt-on-error")
        cmd.append(tex_path.name)
        proc = subprocess.run(cmd, cwd=build_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        combined.append(proc.stdout)
        if proc.returncode != 0:
            log_path.write_text("\n".join(combined), encoding="utf-8", errors="replace")
            raise RuntimeError(f"xelatex failed for {tex_path}. See {log_path}.")

    log_path.write_text("\n".join(combined), encoding="utf-8", errors="replace")
    return NotesBuildResult(
        tex_path=tex_path,
        pdf_path=build_dir / f"{tex_path.stem}.pdf",
        build_dir=build_dir,
        log_path=log_path,
    )


def render_notes_pdf(
    output_dir: str | Path,
    *,
    filename: str = "notes.tex",
    title: str,
    subtitle: str = "",
    description: str = "",
    body: str,
    doc_date: str | None = None,
    compile_pdf: bool = True,
) -> NotesBuildResult:
    """Write and optionally compile a note document in the package style."""
    tex_path = write_notes_tex(
        output_dir,
        filename=filename,
        title=title,
        subtitle=subtitle,
        description=description,
        body=body,
        doc_date=doc_date,
    )
    if compile_pdf:
        return compile_notes_pdf(tex_path)
    return NotesBuildResult(
        tex_path=tex_path,
        pdf_path=tex_path.with_suffix(".pdf"),
        build_dir=tex_path.parent,
        log_path=tex_path.with_suffix(".build.log"),
    )



def build_frontend_manual(
    output_dir: str | Path = "docs/frontend_manual",
    *,
    compile_pdf: bool = True,
) -> NotesBuildResult:
    """Generate the Chinese front-end manual and its example figures."""
    output_dir = Path(output_dir)
    asset_dir = output_dir / "assets"
    figures = generate_frontend_manual_figures(asset_dir)
    tex_figures = {name: Path("assets") / path.name for name, path in figures.items()}
    body = frontend_manual_body(tex_figures)
    return render_notes_pdf(
        output_dir,
        filename="frontend_manual_zh.tex",
        title="Zou_lab_control.frontend 中文手册",
        subtitle="Jupyter / GUI 实验绘图前端",
        description="统一 array 数据契约、live session、fit、selector 与 PDF notes",
        body=body,
        doc_date=date.today().isoformat(),
        compile_pdf=compile_pdf,
    )


__all__ = [
    "NotesBuildResult",
    "build_frontend_manual",
    "compile_notes_pdf",
    "notes_template_dir",
    "render_notes_pdf",
    "write_notes_tex",
]
