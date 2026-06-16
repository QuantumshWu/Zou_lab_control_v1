"""Built-in data-processing actions (auto-discovered by ``processor_registry``).

Drop a module here with an ``@processor`` factory ``build(readout) -> ProcessorSpec``
and it appears in ``exp.readout.processor_specs()`` and the task console's Add-Panel
"Data processing" group with no catalog edit -- exactly as first-class as the
built-ins.  Each processor DRIVES existing analysis (it re-implements no readout /
fidelity math) and publishes named results to the shared SignalHub.
"""
