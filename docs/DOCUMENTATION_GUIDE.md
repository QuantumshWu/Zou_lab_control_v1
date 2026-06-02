# Documentation Guide

This project has two documentation audiences. Keep them separate.

## User Manuals

User manuals are tutorials for people who are new to the project. They should explain:

- what the system is for;
- how to install and start notebooks;
- how the hardware/control layers fit together;
- what each API call does;
- what output to expect;
- how to troubleshoot common failures.

Write manuals in a neutral instructional voice. Avoid conversational or internal-review language.

Good:

```text
capture displays the raw camera frame. Site overlays are shown by sitemap and detect results because those plots use readout calibration.
```

Avoid:

```text
if capture shows sitemap circles, that is a serious architecture error.
```

The second sentence is useful as a maintainer invariant, but it belongs in agent notes or code-review findings, not in a user manual.

## Agent And Maintainer Notes

Agent-facing notes can record implementation constraints and architectural invariants more directly. Put those notes in:

- `AGENTS.md`
- `docs/PROJECT_OVERVIEW.md`
- design-specific Markdown files under `docs/`

These notes may mention anti-patterns, failure modes, and review findings, but should still be concise and actionable.

## Source Of Truth For Generated Docs

Generated manuals should be edited at their templates:

- hardware quickstart body: `Zou_lab_control/neutral_atom/content/manual_templates/hardware_quickstart_zh.texbody`
- frontend manual body: `Zou_lab_control/frontend/content/manual_templates/frontend_manual_zh.texbody`
- notebook templates: `Zou_lab_control/frontend/content/notebook_templates/*.cells.md`

After editing a template, regenerate the checked-in artifact if the repo already tracks it.

## Notebook Text

Notebook markdown should be short and operational:

- explain what the next cell does;
- show the concrete command or API call;
- avoid long architecture essays inside notebooks;
- link or point to manuals for deeper background.

## Style Rules

- Prefer "this component does X" over "do not do Y" unless it is a safety-critical instruction.
- Prefer "recommended path" and "fallback path" over "right/wrong".
- Explain ownership of state: camera, sequencer, session, readout calibration, frontend plot.
- Do not include chat history, blame, or implementation apology in user manuals.
- Keep historical code discussion in migration docs or `references/`, not in quickstarts.
