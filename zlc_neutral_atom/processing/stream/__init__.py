"""One-input exact stream processing baseline."""

from .contract import (
    BoundStreamProcessor,
    JoinKeyTransform,
    ProcessorExecutionGuard,
    StreamCardinality,
    StreamJoinPolicy,
    StreamProcessorDefinition,
)
from .worker import ExactStreamProcessorWorker, StreamProcessorError

__all__ = [
    "BoundStreamProcessor",
    "ExactStreamProcessorWorker",
    "JoinKeyTransform",
    "ProcessorExecutionGuard",
    "StreamCardinality",
    "StreamJoinPolicy",
    "StreamProcessorDefinition",
    "StreamProcessorError",
]
