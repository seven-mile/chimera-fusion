from .proto import (
    ScheduleMethod,
    CellType,
    ScheduleCell,
    PipelineRankStageManager,
    PipelineScheduleManager,
)

from .chimera import (
    ChimeraPipelineRankStageManager,
    ChimeraPipelineScheduleManager,
)

from .ifib import (
    IFIBPipelineRankStageManager,
)

def create_pipeline_rank_stage_manager(
    method: ScheduleMethod,
    num_prs_keys: int,
    num_devices: int,
    num_stages: int,
    world_rank: int,
) -> PipelineRankStageManager:
    if method == ScheduleMethod.CHIMERA:
        return ChimeraPipelineRankStageManager(num_prs_keys, num_devices, num_stages, world_rank)
    elif method == ScheduleMethod.INTERLEAVED:
        raise NotImplementedError
    else:
        return IFIBPipelineRankStageManager(num_devices, num_stages, world_rank)


def create_pipeline_schedule_manager(
    method: ScheduleMethod,
    num_pipelines: int,
    num_devices: int,
    num_stages: int,
    world_rank: int,
    micro_size: int,
) -> PipelineScheduleManager:
    if method == ScheduleMethod.CHIMERA:
        return ChimeraPipelineScheduleManager(num_pipelines, num_devices, num_stages, world_rank, micro_size)
    elif method == ScheduleMethod.INTERLEAVED:
        raise NotImplementedError
    else:
        raise NotImplementedError
