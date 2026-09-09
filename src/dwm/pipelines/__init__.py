"""Model-agnostic thermal video pipeline interfaces and orchestration."""

from .contracts import (
    ConditionBundle,
    GenerationResult,
    PipelineContractError,
    ThermalVideoBatch,
    ThermalVideoModel,
    VideoEvaluator,
    VideoWriter,
)
from .objectives import LossResult, MaskedDiffusionMSE, VideoObjective
from .thermal_video import PipelineStateError, ThermalVideoPipeline

__all__ = [
    "ConditionBundle",
    "GenerationResult",
    "LossResult",
    "MaskedDiffusionMSE",
    "PipelineContractError",
    "PipelineStateError",
    "ThermalVideoBatch",
    "ThermalVideoModel",
    "ThermalVideoPipeline",
    "VideoEvaluator",
    "VideoObjective",
    "VideoWriter",
]
