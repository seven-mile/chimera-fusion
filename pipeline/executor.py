
import torch
from torch.cuda import nvtx
from torch import distributed as dist, nn

from .context import PipelineContext
from .stage import PipelineStage

import comm

from typing import Dict, List, Iterator, Tuple

class PipelineExecutor:

    def __init__(self,
                 pctx: PipelineContext,
                 stages: Dict[int, PipelineStage],
                 data_iters: List[Iterator],
                 num_interations: int):
        """
        Initialize the pipeline executor.
        
        Args:
            pctx (PipelineContext): Pipeline context.
            stages (Dict[int, PipelineStage]): Dict of stage ids and their corresponding pipeline stages.
            data_iters (List[Iterator]): List of data iterators.
            num_interations (int): Number of iterations.
        """
        
        self.pctx = pctx
        self.stages = stages
        self.data_iters = data_iters
        self.num_interations = num_interations

        self.sched = self.pctx.sched_mgr.get_schedule(self.pctx.world_rank)

        self.scms = self._init_scms()

        for stage in self.stages.values():
            stage.stage_module.train()

    def _init_scms(self):

        scms = {}
        for stage in self.stages.values():
            sid = stage.stage_id
            scm = comm.StageCommunicationManager(self.pctx.device)

            def prepend_batch_sizes(shape_dict: Dict[str, Tuple]):
                return {key: self.pctx.batch_sizes + shape for key, shape in shape_dict.items()}

            if self.pctx.is_interleaved:
                tags = [sid, sid+1, sid+1, sid]
            else:
                tags = [stage.p2p_tag] * 4

            scm.start_recv_threads(self.num_interations, stage.forward_recv_queues, stage.prev_rank, prepend_batch_sizes(stage.sizes_from_prev_stage), tags[0])
            scm.start_send_threads(self.num_interations, stage.forward_send_queues, stage.next_rank, tags[1])
            scm.start_recv_threads(self.num_interations, stage.backward_recv_queues, stage.next_rank, prepend_batch_sizes(stage.sizes_for_next_stage), tags[2])
            scm.start_send_threads(self.num_interations, stage.backward_send_queues, stage.prev_rank, tags[3])
            scms[stage.stage_id] = scm

        return scms
    
    def _assert_intermediate_queues_are_empty(self):
        """
        Assert that all intermediate queues are empty.
        """
        for stage in self.stages.values():
            stage.assert_intermediate_queues_are_empty()

    def run(self, iteration: int = None):
        """
        Run the pipeline for one iteration.

        Args:
            iteration (int): Iteration id.

        Returns:
            float: Total loss of the pipeline.
        """

        self._assert_intermediate_queues_are_empty()

        for idx, cell in enumerate(self.sched):
            peek_sync = self.pctx.is_chimera and self.pctx.is_layer_allreduce \
                and idx != len(self.sched) - 1 and self.sched[idx+1].is_sync()
            
            if cell.is_idle():
                continue

            print(f'Z scheduling cell {cell}', flush=True)

            stage = self.stages[cell.stage_id]
            print(f'Z communication prev_rank {stage.prev_rank} next_rank {stage.next_rank}', flush=True)
            if cell.is_sync():
                stage.sync_grad()
            elif cell.is_forward():
                stage.call_forward(next(self.data_iters[cell.prs_key]))
            else:
                if peek_sync:
                    handles = stage.install_sync_hooks()
                stage.call_backward()
                if peek_sync:
                    stage.uninstall_sync_hooks(handles)

        self._assert_intermediate_queues_are_empty()

        # wait for all gradients to be synchronized
        for stage in self.stages.values():
            stage.wait_all()
        
        torch.cuda.synchronize()

        last_stage_id = self.pctx.num_stages - 1
        last_stage = self.stages.get(last_stage_id)
        assert last_stage is None or last_stage.is_last_stage

        return 0.0 if last_stage is None else last_stage.total_loss

