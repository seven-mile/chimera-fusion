

from dataclasses import dataclass
from enum import Enum

import torch
import torch.distributed

from chimera_pipeline_rank import (
    PipelineRankStageManager,
    PipelineScheduleManager,
    ChimeraPipelineRankStageManager,
    DapplePipelineRankStageManager,
    ChimeraScheduleManager,
)
from utils import init_dist_process_group

class PipelineMethod(Enum):
    GPIPE = 'gpipe'
    ONE_F_ONE_B = '1f1b'
    INTERLEAVED = 'interleaved'
    CHIMERA = 'chimera'


class PipelineContext:

    def __init__(self, /, p2p_backend: str, collective_backend: str, pipeline_method: str, num_stages: int, micro_batch_size: int, gradient_accumulation_steps: int, num_pipelines: int, num_chunks: int, recompute: bool, max_seq_length: int):
        _, _, world_rank, world_size = self._init_communication(p2p_backend)

        self.world_size = world_size
        self.world_rank = world_rank
        self.device = torch.device('cuda', torch.cuda.current_device())

        self.pipeline_method = PipelineMethod(pipeline_method)
        self.num_stages = num_stages
        self.micro_batch_size = micro_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_pipelines = num_pipelines
        self.num_chunks = num_chunks
        self.recompute = recompute
        self.max_seq_length = max_seq_length

        self.stage_rank_mgr = self._get_prs_mgr()
        self.sched_mgr = self._get_sched_mgr()

        self._validate()

        self.sync_groups = self._create_groups(collective_backend)
    
    # derived properties
    @property
    def num_prs_keys(self):
        return self.stage_rank_mgr.num_prs_keys
    
    @property
    def is_distriubted(self):
        return self.num_replicas > 1
    
    @property
    def is_master(self):
        return self.world_rank == 0
    
    @property
    def is_chimera(self):
        return self.pipeline_method == PipelineMethod.CHIMERA
    
    @property
    def is_interleaved(self):
        return self.pipeline_method == PipelineMethod.INTERLEAVED

    @property
    def num_ranks_per_stage(self):
        if self.is_interleaved:
            return self.world_size // (self.num_stages // self.num_chunks)
        else:
            return self.world_size // self.num_stages
        
    @property
    def micro_size(self):
        return self.num_stages * self.gradient_accumulation_steps
    
    @property
    def num_micro_batches_per_step(self):
        if self.is_chimera:
            return self.micro_size // self.num_pipelines 
        return self.micro_size
    
    @property
    def num_replicas(self):
        if self.is_chimera:
            return self.num_ranks_per_stage * self.num_pipelines
        else:
            return self.num_ranks_per_stage
    
    @property
    def batch_size(self):
        return self.micro_batch_size

    @property
    def batch_sizes(self):
        return (self.batch_size, self.max_seq_length)

    def _validate(self):
        assert self.world_size > 0
        assert self.world_rank >= 0
        assert self.num_replicas > 0
        assert self.pipeline_method in PipelineMethod
        assert self.num_stages > 0
        assert self.micro_batch_size > 0
        assert self.stage_rank_mgr is not None
        assert self.sched_mgr is not None

        if self.is_interleaved:
            assert self.num_chunks > 1
            assert self.num_stages % self.num_chunks == 0
            assert self.world_size % (self.num_stages // self.num_chunks) == 0
        else:
            if self.is_chimera:
                assert self.num_pipelines > 0
                assert self.num_pipelines > 1, "num_pipelines should be greater than 1 for chimera pipelines"
            assert self.world_size % self.num_stages == 0
    
    def _init_communication(self, p2p_backend: str):
        local_rank, local_size, world_rank, world_size = init_dist_process_group(backend=p2p_backend)
        assert local_size <= torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        assert world_size > 1
        return local_rank, local_size, world_rank, world_size
    
    def _create_groups(self, collective_backend: str):
        torch.distributed.barrier()
        return [
            torch.distributed.new_group(ranks, backend=collective_backend)
            for ranks in self.stage_rank_mgr.get_stage_to_ranks_map()
        ]

    def _get_prs_mgr(self) -> PipelineRankStageManager:
        if self.is_chimera:
            return ChimeraPipelineRankStageManager(self.num_pipelines, self.world_size, self.num_stages, self.world_rank)
        elif self.is_interleaved:
            raise NotImplementedError
        else:
            return DapplePipelineRankStageManager(self.world_size, self.num_stages, self.world_rank)
    
    def _get_sched_mgr(self) -> PipelineScheduleManager:
        if self.is_chimera:
            return ChimeraScheduleManager(self.num_pipelines, self.world_size, self.num_stages, self.world_rank, self.micro_size)
        else:
            raise NotImplementedError
