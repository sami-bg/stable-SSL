from .checkpoint_sklearn import SklearnCheckpoint, WandbCheckpoint
from .trainer_info import LoggingCallback, ModuleSummary, TrainerInfo, SLURMInfo
from .env_info import EnvironmentDumpCallback
from .registry import ModuleRegistryCallback
from .unused_parameters import LogUnusedParametersOnce
from .cpu_offload import CPUOffloadCallback
from .hf_models import HuggingFaceCheckpointCallback

def default():
    """Factory function that returns default callbacks."""
    callbacks = [
        # RichProgressBar(),
        ModuleRegistryCallback(),
        LoggingCallback(),
        EnvironmentDumpCallback(async_dump=True),
        TrainerInfo(),
        SklearnCheckpoint(),
        WandbCheckpoint(),
        ModuleSummary(),
        SLURMInfo(),
        LogUnusedParametersOnce(),
        CPUOffloadCallback(),
        HuggingFaceCheckpointCallback()
    ]

    return callbacks
