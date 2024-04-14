from .proto import (
    ScheduleMethod,
    GradReduceMethod,
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
    IFIBPipelineScheduleManager,
)

from .interleaved import (
    InterleavedPipelineRankStageManager,
    InterleavedPipelineScheduleManager,
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
        return InterleavedPipelineRankStageManager(num_prs_keys, num_devices, num_stages, world_rank)
    elif method == ScheduleMethod.IFIB:
        return IFIBPipelineRankStageManager(num_devices, num_stages, world_rank)
    else:
        raise NotImplementedError


def create_pipeline_schedule_manager(
    method: ScheduleMethod,
    num_prs_keys: int,
    num_devices: int,
    num_stages: int,
    world_rank: int,
    micro_size: int,
    grad_reduce_method: GradReduceMethod = GradReduceMethod.BASELINE,
) -> PipelineScheduleManager:
    if method == ScheduleMethod.CHIMERA:
        return ChimeraPipelineScheduleManager(num_prs_keys, num_devices, num_stages, world_rank, micro_size, grad_reduce_method)
    elif method == ScheduleMethod.INTERLEAVED:
        return InterleavedPipelineScheduleManager(num_prs_keys, num_devices, num_stages, world_rank, micro_size)
    elif method == ScheduleMethod.IFIB:
        return IFIBPipelineScheduleManager(num_devices, num_stages, world_rank, micro_size)
    else:
        raise NotImplementedError
