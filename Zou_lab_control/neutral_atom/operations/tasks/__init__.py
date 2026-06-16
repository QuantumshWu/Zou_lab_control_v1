"""Built-in orchestration TASKS (auto-discovered by ``task_registry``).

Drop a module here with a ``@task`` factory ``build(readout) -> TaskSpec`` and it
appears in ``exp.readout.task_specs()`` and the task console's Add-Panel "Task"
group with no catalog edit -- as first-class as the built-ins.  Each task builds a
:class:`~..feeds.Task` over the session (capturing the readout subsystem) that
streams mid-run output to its own dedicated panel and derives an artifact.
"""
