from .proto import CellType, PipelineRankStageManager, PipelineScheduleManager, ScheduleCell

from typing import List

from collections import deque

class InterleavedPipelineRankStageManager(PipelineRankStageManager):
    """
    PipelineRankStageManager for Interleaved 1F1B pipeline scheduling.

    The mapping is a simple extension to classical 1F1b:
    - the number of stages is some multiple of the number of devices, which is called chunk
    - stages from different chunks are folded to the same device
    """
    def __init__(self, num_chunks: int, num_devices: int, num_stages: int, rank: int):
        """
        Initialize the InterleavedPipelineRankStageManager.

        Args:
            num_devices (int): The number of devices.
            num_stages (int): The number of stages.
            rank (int): The rank of the current process.
        """

        if num_devices <= 0 or num_stages <= 0:
            raise ValueError("The number of devices and stages should be positive integers")
        
        if rank < 0 or rank >= num_devices:
            raise ValueError("The rank of the current process should be in the range of [0, num_devices)")
        
        if num_stages % num_chunks != 0:
            raise ValueError("The number of stages should be a multiple of the number of chunks")

        if num_devices % (num_stages // num_chunks) != 0:
            raise ValueError("The number of devices should be a multiple of the number of stages per chunk")

        self._num_chunks = num_chunks
        self._num_devices = num_devices
        self._num_stages = num_stages
        self._this_rank = rank
        self._construct()

    @property
    def num_prs_keys(self):
        return self._num_chunks
    
    @property
    def num_stages(self):
        return self._num_stages
    
    @property
    def num_devices(self):
        return self._num_devices

    def get_rank_to_stage_map(self, chunk_id: int) -> List[int]:
        """
        Get the rank to stage map of the pipeline with the given index.

        Args:
            chunk_id (int): The index of the chunk.

        Returns:
            List[int]: The rank to stage map.
        """
        return self._rank_to_stage_map[chunk_id]
    
    def get_stage_to_rank_map(self, _=None) -> List[int]:
        """
        Get the stage to rank map of the pipeline with the given index.

        Returns:
            List[int]: The stage to rank map.
        """
        return self._stage_to_rank_map
    
    def _construct(self):
        self._stage_to_rank_map = [-1 for _ in range(self._num_stages)]
        self._rank_to_stage_map = [[-1 for _ in range(self._num_devices)] for _ in range(self._num_chunks)]

        per_chunk_stages = self._num_stages // self._num_chunks

        per_stage_devices = self._num_devices // per_chunk_stages
        this_local_rank = self._this_rank % per_stage_devices

        for stage_id in range(self._num_stages):
            chunk_id = stage_id // per_chunk_stages
            base_rank = (stage_id % per_chunk_stages) * per_stage_devices
            self._stage_to_rank_map[stage_id] = base_rank + this_local_rank
            for local_rank in range(per_stage_devices):
                self._rank_to_stage_map[chunk_id][base_rank + local_rank] = stage_id

