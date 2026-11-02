"""Task-internal algorithms for marl_platoon.

These modules are imported by the IsaacLab task and do not define a standalone
training entrypoint.
"""

from .router import PlatoonAlgorithmRouter, build_platoon_algorithm_router
from .student import HAPPOStudentModule
from .teacher import RewardTeacherCfg, RewardTeacherModule
from .attacker import AttackModuleCfg, NoOpAttackModule
from .shield import BadHeadingShieldCfg, BadHeadingShieldModule
from .pipeline import (
    PipelineFlags,
    PipelineScheduleCfg,
    PlatoonTrainingPipeline,
    StepBatch,
    StudentModule,
    TeacherModule,
    AttackerModule,
    ShieldModule,
)

__all__ = [
    "PlatoonAlgorithmRouter",
    "build_platoon_algorithm_router",
    "PipelineFlags",
    "PipelineScheduleCfg",
    "PlatoonTrainingPipeline",
    "StepBatch",
    "StudentModule",
    "TeacherModule",
    "AttackerModule",
    "ShieldModule",
    "HAPPOStudentModule",
    "RewardTeacherCfg",
    "RewardTeacherModule",
    "AttackModuleCfg",
    "NoOpAttackModule",
    "BadHeadingShieldCfg",
    "BadHeadingShieldModule",
]
