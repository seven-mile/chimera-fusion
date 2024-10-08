
import pytest

import sys

from . import (
    ScheduleMethod,
    create_pipeline_rank_stage_manager,
    create_pipeline_schedule_manager,
    ChimeraPipelineScheduleManager,
    IFIBPipelineScheduleManager,
    InterleavedPipelineScheduleManager,
)


chimera_prs_tests = [
    (ScheduleMethod.CHIMERA, 2, 4, 0, 8),
    (ScheduleMethod.CHIMERA, 2, 4, 1, 8),
    (ScheduleMethod.CHIMERA, 2, 4, 0, 16),
    (ScheduleMethod.CHIMERA, 4, 8, 0, 16),
    (ScheduleMethod.IFIB, 1, 8, 0, 8),
    (ScheduleMethod.IFIB, 1, 8, 0, 16),
    (ScheduleMethod.IFIB, 1, 8, 7, 16),
    (ScheduleMethod.INTERLEAVED, 2, 16, 0, 8),
]

@pytest.mark.parametrize("method, num_prs_keys, num_stages, this_rank, num_devices", chimera_prs_tests)
def test_prs(method, num_prs_keys, num_stages, this_rank, num_devices):

    num_devices_per_stage = num_prs_keys * num_devices // num_stages
    
    mgr = create_pipeline_rank_stage_manager(method, num_prs_keys, num_devices, num_stages, this_rank)
    assert len(mgr.get_stage_to_ranks_map()) == num_stages, "The number of stages should be equal to the length of the stage to ranks map"
    
    assert all(len(mgr.get_stage_to_ranks_map()[i]) == num_devices_per_stage for i in range(num_stages)), \
        "The number of ranks in the stage to ranks map should be equal to the number of devices per stage"

    for key in range(num_prs_keys):
        s2r = mgr.get_stage_to_rank_map(key)
        r2s = mgr.get_rank_to_stage_map(key)
        assert len(s2r) == num_stages, "The number of stages should be equal to the length of the stage to ranks map"
        assert len(r2s) == num_devices, "The number of devices should be equal to the length of the rank to stage map"
        for i in range(num_stages):
            mgr.get_stage_to_rank_map(key)[i]


chimera_sched_tests = [
    (2, 4, 4, 0, 8),
    (4, 4, 4, 0, 8),
    (2, 4, 8, 0, 8),
    (4, 4, 8, 0, 8),
    (2, 6, 6, 0, 12),
]

@pytest.mark.parametrize("num_pipelines, num_stages, micro_size, this_rank, num_devices", chimera_sched_tests)
def test_chimera_schedule(num_pipelines, num_stages, micro_size, this_rank, num_devices):
    mgr: ChimeraPipelineScheduleManager = create_pipeline_schedule_manager(ScheduleMethod.CHIMERA, num_pipelines, num_devices, num_stages, this_rank, micro_size)
    assert isinstance(mgr, ChimeraPipelineScheduleManager)
    
    assert len(mgr._schedule) == num_stages, "The number of stages should be equal to the length of the first dim of schedule"
    
    print(mgr, flush=True)


@pytest.mark.parametrize("num_stages, micro_size, this_rank, num_devices", [(8, 8, 0, 8)])
def test_1f1b_schedule(num_stages, micro_size, this_rank, num_devices):
    mgr: IFIBPipelineScheduleManager = create_pipeline_schedule_manager(ScheduleMethod.IFIB, 1, num_devices, num_stages, this_rank, micro_size)
    assert isinstance(mgr, IFIBPipelineScheduleManager)
    
    # TODO: Add more assertions
    print(mgr, flush=True)


@pytest.mark.parametrize("num_chunks, num_stages, micro_size, this_rank, num_devices", [(2, 8, 8, 0, 8)])
def test_interleaved_schedule(num_chunks, num_stages, micro_size, this_rank, num_devices):
    mgr: InterleavedPipelineScheduleManager = create_pipeline_schedule_manager(ScheduleMethod.INTERLEAVED, num_chunks, num_devices, num_stages, this_rank, micro_size)
    assert isinstance(mgr, InterleavedPipelineScheduleManager)
    
    # TODO: Add more assertions
    print(mgr, flush=True)
