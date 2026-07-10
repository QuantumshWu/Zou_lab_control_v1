"""XeLaTeX notes/manual generation for the Zou lab front-end."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from importlib import resources
from pathlib import Path
import shutil
import subprocess
import tempfile

from .content.manuals import frontend_manual_body, generate_frontend_manual_figures


@dataclass(frozen=True)
class NotesBuildResult:
    """Paths produced by a notes/manual build."""

    tex_path: Path
    pdf_path: Path
    build_dir: Path
    log_path: Path | None


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


def build_notes_tex_string(
    *,
    title: str,
    subtitle: str = "",
    description: str = "",
    body: str,
    doc_date: str | None = None,
    template_package: str = "zlc_frontend_notes",
) -> str:
    """Assemble a complete notes document as a TeX STRING (no filesystem writes).

    This is the single source of the document wrapper (title page + TOC + body);
    callers compile it through :func:`render_tex_pdf`, so the template structure
    is guaranteed and no intermediates are written to the output directory."""

    doc_date = doc_date or date.today().isoformat()
    title_tex = _latex_text(title)
    subtitle_tex = _latex_text(subtitle)
    description_tex = _latex_text(description)
    date_tex = _latex_text(doc_date)
    return rf"""\documentclass[11pt,oneside,openany]{{book}}
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
    """Write an inspectable XeLaTeX source bundle (leaves a ``.tex`` + ``.sty``).

    The default manual build does NOT use this -- it assembles the tex in memory
    via :func:`build_notes_tex_string` and compiles through :func:`render_tex_pdf`
    so the output directory keeps only the PDF.  Use this only when you
    deliberately want the source ``.tex`` on disk for inspection."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_template_files(output_dir)
    tex_path = output_dir / filename
    tex = build_notes_tex_string(
        title=title, subtitle=subtitle, description=description, body=body,
        doc_date=doc_date, template_package=template_package)
    tex_path.write_text(tex, encoding="utf-8")
    return tex_path


def render_tex_pdf(
    tex: str | Path,
    output_pdf: str | Path,
    *,
    runs: int = 2,
    xelatex: str | None = None,
    halt_on_error: bool = True,
    assets: str | Path | None = None,
) -> Path:
    """Compile TeX to PDF without leaving auxiliary files beside the output.

    ``tex`` may be a complete TeX string or a path to a ``.tex`` file.  When a
    file path is supplied, its sibling files and folders are copied to a
    temporary build directory so relative assets such as ``assets/foo.pdf`` keep
    working.  When a STRING is supplied and ``assets`` names a directory, that
    directory is copied into the temp build dir as ``assets/`` so figures
    referenced as ``assets/foo.png`` resolve.  On success, ONLY ``output_pdf`` is
    copied back -- nothing else is written anywhere.  On failure, a ``.build.log``
    next to ``output_pdf`` records the XeLaTeX output.
    """

    output_pdf = Path(output_pdf).resolve()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_pdf.with_suffix(".build.log")
    output_pdf.unlink(missing_ok=True)
    log_path.unlink(missing_ok=True)
    exe = xelatex or shutil.which("xelatex")
    if exe is None:
        message = (
            "xelatex was not found on PATH.\n"
            "Install a TeX distribution with XeLaTeX, or pass xelatex='C:/path/to/xelatex'.\n"
        )
        log_path.write_text(message, encoding="utf-8", errors="replace")
        raise RuntimeError(f"xelatex was not found on PATH. See {log_path}.")

    source_path = _existing_tex_path(tex)
    with tempfile.TemporaryDirectory(prefix="zlc_latex_") as tmp:
        build_dir = Path(tmp)
        _copy_template_files(build_dir)
        if source_path is not None:
            source_path = source_path.resolve()
            skip_rel = {source_path.with_suffix(".pdf").relative_to(source_path.parent)}
            try:
                skip_rel.add(output_pdf.relative_to(source_path.parent))
            except ValueError:
                pass
            _copy_latex_source_tree(source_path.parent, build_dir, skip_paths=skip_rel)
            tex_name = source_path.name
        else:
            tex_name = "document.tex"
            (build_dir / tex_name).write_text(str(tex), encoding="utf-8")
            if assets is not None:
                asset_dir = Path(assets)
                if asset_dir.is_dir():
                    shutil.copytree(asset_dir, build_dir / "assets", dirs_exist_ok=True)

        combined: list[str] = []
        for _ in range(max(1, int(runs))):
            cmd = [exe, "-interaction=nonstopmode"]
            if halt_on_error:
                cmd.append("-halt-on-error")
            cmd.append(tex_name)
            try:
                proc = subprocess.run(cmd, cwd=build_dir, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            except FileNotFoundError as exc:
                log_path.write_text(f"xelatex executable was not found: {exe}\n{exc}\n", encoding="utf-8", errors="replace")
                raise RuntimeError(f"xelatex executable was not found for {tex_name}. See {log_path}.") from exc
            combined.append(proc.stdout)
            if proc.returncode != 0:
                log_path.write_text("\n".join(combined), encoding="utf-8", errors="replace")
                raise RuntimeError(f"xelatex failed for {tex_name}. See {log_path}.")

        built_pdf = build_dir / f"{Path(tex_name).stem}.pdf"
        if not built_pdf.exists():
            raise RuntimeError(f"xelatex completed but did not produce {built_pdf.name}.")
        shutil.copy2(built_pdf, output_pdf)
    return output_pdf


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
    runs: int = 2,
    xelatex: str | None = None,
    halt_on_error: bool = True,
) -> NotesBuildResult:
    """Build a manual PDF, leaving ONLY the ``.pdf`` in ``output_dir``.

    The document is assembled in memory (:func:`build_notes_tex_string`) and
    compiled through :func:`render_tex_pdf` in a temporary directory, so no
    ``.tex``/``.sty``/``.aux``/``.log``/``.toc``/``.out`` is ever written to
    ``output_dir`` -- only the final ``<filename stem>.pdf`` (alongside any
    committed ``assets/`` it embeds).  ``compile_pdf=False`` is an explicit
    inspection mode that writes the source ``.tex`` bundle."""
    output_dir = Path(output_dir)
    pdf_path = output_dir / f"{Path(filename).stem}.pdf"
    if not compile_pdf:
        tex_path = write_notes_tex(
            output_dir, filename=filename, title=title, subtitle=subtitle,
            description=description, body=body, doc_date=doc_date)
        return NotesBuildResult(tex_path=tex_path, pdf_path=pdf_path,
                                build_dir=output_dir, log_path=None)
    tex = build_notes_tex_string(
        title=title, subtitle=subtitle, description=description, body=body,
        doc_date=doc_date)
    asset_dir = output_dir / "assets"
    pdf_path = render_tex_pdf(
        tex, pdf_path, runs=runs, xelatex=xelatex, halt_on_error=halt_on_error,
        assets=asset_dir if asset_dir.is_dir() else None)
    return NotesBuildResult(
        tex_path=output_dir / filename,   # nominal source path; NOT written on the clean path
        pdf_path=pdf_path,
        build_dir=output_dir,
        log_path=None,
    )


def _copy_latex_source_tree(source_dir: Path, build_dir: Path, *, skip_paths: set[Path] | None = None) -> None:
    skip_paths = {Path(path) for path in (skip_paths or set())}
    skip_suffixes = {
        ".aux",
        ".bbl",
        ".bcf",
        ".blg",
        ".fdb_latexmk",
        ".fls",
        ".log",
        ".lof",
        ".lot",
        ".out",
        ".run.xml",
        ".synctex.gz",
        ".toc",
    }
    for src in source_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(source_dir)
        if rel in skip_paths:
            continue
        suffix = "".join(src.suffixes[-2:]) if src.name.endswith(".synctex.gz") else src.suffix
        if suffix in skip_suffixes:
            continue
        dest = build_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _existing_tex_path(tex: str | Path) -> Path | None:
    if isinstance(tex, Path):
        path = tex
    else:
        text = str(tex)
        if "\n" in text or "\\documentclass" in text:
            return None
        try:
            path = Path(text)
        except (OSError, ValueError):
            return None
    try:
        if path.suffix.lower() == ".tex" and path.exists():
            return path
    except OSError:
        return None
    return None



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
    "notes_template_dir",
    "render_tex_pdf",
    "render_notes_pdf",
    "write_notes_tex",
]
