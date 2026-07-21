"""The saved-layout record lands in zlc_frontend -- the question the last two slices deferred.

S5-shell(u) proved ``TaskConsoleState`` cannot live in ``zlc_data``: it carries
``save(path)`` / ``load(path)``, and L303 admits only "可序列化的数据语义和值上的纯算法"
while L320 forbids repository/IO there.  That left the real question open, and the answer
is also in the document rather than in taste:

  L315  ``zlc_storage`` 只拥有两类窄基础设施 ... 它不定义 universal ArtifactRef、
        领域 schema 或 artifact kind
  L327  zlc_frontend -> zlc_storage

A console-layout reader CONSTRUCTS a domain record, so it is not storage's to own; and the
frontend is explicitly allowed to depend on storage.  So the record lives in
``zlc_frontend.console_state``, its vocabulary and persisted schema stay in
``zlc_data.console_records``, and the folder it anchors to comes from ``zlc_storage.paths``.

Reconnaissance also settled the durability question by measurement rather than assumption:
the repository contains exactly ONE atomic publish (``os.replace`` inside
``ContentAddressedStore``), and it is a content-addressed blob store, not a
write-this-file-atomically utility.  Making layout writes crash-safe therefore means
DESIGNING a storage primitive, which is not a move.  The gap is recorded, and
``test_saving_is_still_an_in_place_write`` states the current behaviour honestly so the
day someone fixes it, they change an assertion that describes the world rather than
discovering a silent regression.

The frozen tests covering this (test_frontend_smoke.py, test_panel_view_logic_split.py)
sit outside ``tests/migration_active_tests.txt``, so the cases below were captured from the
shell immediately before the move.
"""

from __future__ import annotations

#: C41 -- these specific tests guard legacy artifacts and die with them; the rest of the
#: file guards the NEW structure and is permanent (swept by test_design_charter).

import ast
import json
import pathlib

import pytest

from zlc_data.console_records import CONSOLE_STATE_SCHEMA, LogicNodeConfig, PanelConfig
from zlc_frontend.console_state import (
    TASK_FILES_ENV,
    TaskConsoleState,
    default_console_state,
    resolve_task_state,
    task_files_dir,
)

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def layouts(tmp_path, monkeypatch):
    monkeypatch.setenv(TASK_FILES_ENV, str(tmp_path / "tasks"))
    return tmp_path


def _state():
    return TaskConsoleState(
        name="rb87", interval_ms=200,
        panels=[PanelConfig(kind="1d", inputs=["cam/frame"], title="P #1")],
        logic=[LogicNodeConfig(kind="processor", name="Analysis", title="A #1")])


def test_an_empty_layout_is_the_captured_default():
    assert default_console_state().to_dict() == {
        "schema": CONSOLE_STATE_SCHEMA, "name": "task", "interval_ms": 400,
        "panels": [], "logic": []}


def test_the_file_keeps_its_key_order_and_stays_readable(layouts):
    """Key ORDER and ``indent=2`` are what make a saved layout diffable and hand-editable;
    both are properties of the file, not of the record, so only reading the file shows them."""

    path = layouts / "rb87.json"
    assert _state().save(path) == path
    text = path.read_text(encoding="utf-8")
    assert list(json.loads(text)) == ["schema", "name", "interval_ms", "panels", "logic"]
    assert text.splitlines()[1].startswith('  "')


def test_a_layout_round_trips_through_a_real_file(layouts):
    path = layouts / "rb87.json"
    state = _state()
    state.save(path)
    assert TaskConsoleState.load(path).to_dict() == state.to_dict()


def test_a_name_resolves_to_the_saved_layout_and_a_path_to_itself(layouts):
    state = _state()
    state.save(task_files_dir() / "my_layout.json")
    elsewhere = layouts / "elsewhere.json"
    state.save(elsewhere)
    assert resolve_task_state("my_layout").name == "rb87"
    assert resolve_task_state(str(elsewhere)).name == "rb87"
    # A human typing a name into the GUI leaves whitespace; the resolver is a named
    # input adapter and strips it (registered in the .strip() ledger for that reason).
    assert resolve_task_state("  my_layout  ").name == "rb87"


@pytest.mark.parametrize("task", ["nope", "missing.json"])
def test_an_unknown_layout_raises_and_names_the_folder_it_searched(task, layouts):
    with pytest.raises(ValueError) as caught:
        resolve_task_state(task if task == "nope" else str(layouts / task))
    assert str(task_files_dir()) in str(caught.value)


@pytest.mark.parametrize("write, error", [
    (None, FileNotFoundError),                       # no file at all
    ("{not json", json.JSONDecodeError),             # not even JSON
    ('{"schema": "other"}', ValueError),             # JSON, wrong format
])
def test_loading_refuses_rather_than_returning_something_empty(write, error, layouts):
    """Four distinct failures, each surfacing as itself: a caller can tell 'you have no
    layout' from 'your file is corrupt' from 'that is not a layout'."""

    path = layouts / "x.json"
    if write is not None:
        path.write_text(write, encoding="utf-8")
    with pytest.raises(error):
        TaskConsoleState.load(path)


def test_saving_is_still_an_in_place_write(layouts):
    """Honest, not aspirational.  ``save`` writes straight to the target, so a crash
    mid-write leaves a truncated layout.  The only atomic publish in the repository lives
    inside ContentAddressedStore for content-addressed blobs; making this crash-safe means
    designing a new storage primitive, which the move deliberately did not smuggle in."""

    source = ast.parse((REPO / "zlc_frontend" / "console_state.py").read_text(encoding="utf-8"))
    save = next(node for node in ast.walk(source)
                if isinstance(node, ast.FunctionDef) and node.name == "save")
    dumped = ast.dump(save)
    assert "write_text" in dumped
    assert "replace" not in dumped, "save now stages+renames -- update this test and the ledger"


def test_the_module_reaches_for_no_renderer_and_no_toolkit():
    """It is a frontend leaf, so Matplotlib and Qt are exactly what it must not pull in."""

    tree = ast.parse((REPO / "zlc_frontend" / "console_state.py").read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    roots = {name.split(".")[0] for name in modules}
    assert not roots & {"matplotlib", "PyQt5", "numpy"}, roots
    assert roots <= {"__future__", "copy", "json", "os", "pathlib", "typing",
                     "zlc_data", "zlc_storage"}, roots




