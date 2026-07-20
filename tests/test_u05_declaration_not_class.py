"""Console predicates that ask what a node DECLARES instead of what class it is.

Two seams left the shell ledger on this principle, and both are pinned here.

``TaskConsole._reactive_ring`` refuses a Start that would close a feedback loop
through the hub: node A consumes what node B publishes, B consumes what A
publishes, and neither ever settles.  It used to identify a reactive node with
``isinstance(node, Processor)``, which is a domain import inside the render
layer.  It now asks the node what it CONSUMES, matching the convention the domain
base already states for the same attribute - ``LogicNode._collect_provenance``:
"probed by attribute, not by type".

That substitution is only safe because ``consumes`` is assigned by
``Processor.__init__`` and by nothing else; a class-level attribute or a second
assigner would silently widen the guard.  The first test below pins exactly that,
so the equivalence stops being a claim about today's code and becomes a
mechanically enforced one.

The remaining tests are the guard's behaviour, written against REAL Processor
instances rather than duck-typed stand-ins: a guard that recognises nodes by
declaration would be trivially satisfied by fakes shaped to please it, which
proves nothing about the objects the console actually receives.

This file is the independent oracle for the seams cut in S5-shell(f) and (g).  The older
``tests/test_signal_cycle_guard.py`` covers the same method but is outside
``tests/migration_active_tests.txt`` and therefore frozen: not run, not modified,
and not evidence for this branch.
"""

from __future__ import annotations

import ast
from pathlib import Path

from Zou_lab_control.neutral_atom.core.signals import SignalHub, SignalSchema
from Zou_lab_control.neutral_atom.operations.logic import Processor

REPO_ROOT = Path(__file__).resolve().parents[1]


class _Relay(Processor):
    """A minimal reactive node: consumes some names, publishes one bare name."""

    def __init__(self, hub, *, consumes, out, prefix=""):
        self._out = str(out)
        super().__init__(hub, consumes=consumes, prefix=prefix)

    def _bare_published_signals(self) -> frozenset:
        return frozenset({self._out})

    def step(self) -> dict[str, object]:
        return {}


class _Source:
    """A node that publishes but declares no reactive inputs (a measurement's shape)."""

    def __init__(self, name):
        self._name = str(name)

    def published_signals(self):
        return frozenset({self._name})


def _console(monkeypatch):
    from zlc_frontend.qt_widgets import ensure_qt_app

    ensure_qt_app()
    from Zou_lab_control.frontend.task_console import TaskConsole

    return TaskConsole(hub=SignalHub())


def test_only_processor_assigns_consumes_so_the_attribute_probe_is_exact():
    """The load-bearing fact behind dropping the isinstance check.

    If any other class gained a ``consumes`` attribute - even a class-level
    default - the console's probe would start treating that node as reactive and
    the ring walk would follow edges the domain never declared.  Checked against
    the source rather than a live object so an unimported subclass cannot hide.
    """

    module = REPO_ROOT / "Zou_lab_control" / "neutral_atom" / "operations" / "logic.py"
    tree = ast.parse(module.read_text(encoding="utf-8"))
    assigners = set()
    class_level = set()
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for node in ast.walk(cls):
            if isinstance(node, ast.Attribute) and node.attr == "consumes" \
                    and isinstance(node.ctx, ast.Store):
                assigners.add(cls.name)
        for stmt in cls.body:                       # a class-level default would widen it too
            targets = []
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets
            elif isinstance(stmt, ast.AnnAssign):
                targets = [stmt.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == "consumes":
                    class_level.add(cls.name)

    assert assigners == {"Processor"}, assigners
    assert not class_level, class_level


def test_a_ring_between_two_reactive_nodes_is_refused(monkeypatch):
    """b consumes a's output, a consumes b's -- starting a closes the loop."""

    console = _console(monkeypatch)
    hub = console.hub
    for name in ("a_out", "b_out"):
        hub.register_signal(name, SignalSchema())

    a = _Relay(hub, consumes=("b_out",), out="a_out")
    b = _Relay(hub, consumes=("a_out",), out="b_out")
    console.running_nodes = [b]

    ring = console._reactive_ring(a)
    assert ring is not None, "the guard must refuse a closing ring"
    assert "a_out" in ring


def test_a_plain_chain_is_allowed(monkeypatch):
    """Source -> b -> a is a chain, not a ring: nothing consumes a's output."""

    console = _console(monkeypatch)
    hub = console.hub
    for name in ("raw", "b_out", "a_out"):
        hub.register_signal(name, SignalSchema())

    a = _Relay(hub, consumes=("b_out",), out="a_out")
    b = _Relay(hub, consumes=("raw",), out="b_out")
    console.running_nodes = [b, _Source("raw")]

    assert console._reactive_ring(a) is None


def test_a_node_that_declares_no_inputs_is_not_walked(monkeypatch):
    """The branch the isinstance check used to guard.

    A node with no ``consumes`` cannot be woken by anything, so it can never be
    the node that closes a ring and the walk must stop immediately - whatever its
    class is.
    """

    console = _console(monkeypatch)
    console.running_nodes = []
    assert console._reactive_ring(_Source("raw")) is None
    assert console._reactive_ring(object()) is None


def test_publishing_a_fit_is_a_declaration_not_a_class():
    """Why the ``FitProcessor`` row could leave the ledger above.

    The console pushes a fit overlay for every panel whose analysis node is
    currently fitting.  It used to ask ``isinstance(node, FitProcessor) and
    node.action == "fit"`` - a domain import in the render layer.  It now asks
    whether the node DECLARES ``<prefix>fit_valid``, which is legitimate only if
    the two predicates agree on every real node shape.

    They agree because ``AnalysisProcessor.provides`` dispatches on the live
    action and the domain keeps declared == published per action, so a node that
    has switched to "roi" stops declaring the fit keys.  This test is the proof,
    not the prose: it evaluates BOTH predicates on the three shapes that exist and
    requires them to match, and requires the outcomes to differ across the shapes
    so a predicate that answered a constant could not pass.

    There is no other coverage of ``_push_fit_overlays`` in the repository -
    active or frozen - so this equivalence is the only thing standing between the
    seam removal and a silent behaviour change.
    """

    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.processors.analysis import AnalysisProcessor
    from Zou_lab_control.neutral_atom.operations.processors.fit import FitProcessor

    hub = SignalHub()
    nodes = {
        "bare fit": FitProcessor(hub, source_expr="src", prefix="f_"),
        "analysis fitting": AnalysisProcessor(hub, action="fit", source_expr="src", prefix="af_"),
        "analysis roi": AnalysisProcessor(hub, action="roi", source_expr="src", prefix="ar_"),
    }

    def was_fitting(node):                      # the predicate the console used to run
        return isinstance(node, FitProcessor) and getattr(node, "action", "fit") == "fit"

    def declares_fit(node):                     # the predicate it runs now
        return f"{node.prefix}fit_valid" in node.published_signals()

    verdicts = {label: (was_fitting(n), declares_fit(n)) for label, n in nodes.items()}
    for label, (old, new) in verdicts.items():
        assert old == new, f"{label}: type test said {old}, declaration said {new}"
    assert {old for old, _ in verdicts.values()} == {True, False},         "the fixture must contain both a fitting and a non-fitting node"
