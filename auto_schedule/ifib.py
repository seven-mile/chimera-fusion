from .proto import CellType, PipelineRankStageManager, PipelineScheduleManager, ScheduleCell

from typing import List

from collections import deque

class IFIBPipelineRankStageManager(PipelineRankStageManager):
    """
    PipelineRankStageManager for 1F1B pipeline scheduling.

    The mapping is straightforward:
    - stage index is rank id
    - rank id is the corresponding rank among the devices of the pipeline stage
    """
    def __init__(self, num_devices: int, num_stages: int, rank: int):
        """
        Initialize the IFIBPipelineRankStageManager.

        Args:
            num_devices (int): The number of devices.
            num_stages (int): The number of stages.
            rank (int): The rank of the current process.
        """

        if num_devices <= 0 or num_stages <= 0:
            raise ValueError("The number of devices and stages should be positive integers")
        
        if rank < 0 or rank >= num_devices:
            raise ValueError("The rank of the current process should be in the range of [0, num_devices)")
        
        if num_devices % num_stages != 0:
            raise ValueError("The number of devices should be a multiple of the number of stages")

        self._num_devices = num_devices
        self._num_stages = num_stages
        self._this_rank = rank
        self._construct()

    @property
    def num_prs_keys(self):
        return 1
    
    @property
    def num_stages(self):
        return self._num_stages
    
    @property
    def num_devices(self):
        return self._num_devices

    def get_rank_to_stage_map(self, _=None) -> List[int]:
        """
        Get the rank to stage map of the pipeline with the given index.

        Returns:
            List[int]: The rank to stage map.
        """
        return self._rank_to_stage_map
    
    def get_stage_to_rank_map(self, _=None) -> List[int]:
        """
        Get the stage to rank map of the pipeline with the given index.

        Returns:
            List[int]: The stage to rank map.
        """
        return self._stage_to_rank_map
    
    def _construct(self):
        self._stage_to_rank_map = [-1 for _ in range(self._num_stages)]
        self._rank_to_stage_map = [-1 for _ in range(self._num_devices)]

        per_stage_devices = self._num_devices // self._num_stages
        this_local_rank = self._this_rank % per_stage_devices

        for stage_id in range(self._num_stages):
            base_rank = stage_id * per_stage_devices
            self._stage_to_rank_map[stage_id] = base_rank + this_local_rank
            for local_rank in range(per_stage_devices):
                self._rank_to_stage_map[base_rank + local_rank] = stage_id


class IFIBPipelineScheduleManager(PipelineScheduleManager):
    """
    PipelineScheduleManager for 1F1B pipeline scheduling.

    The pipeline schedule is a list of time steps, where each step is a list of cells for the corresponding ranks.
    """
    def __init__(self, num_devices: int, num_stages: int, rank: int, micro_size: int):
        """
        Initialize the IFIBPipelineScheduleManager.

        Args:
            num_devices (int): The number of devices.
            num_stages (int): The number of stages.
            ranks (List[int]): The ranks of the devices.
        """

        self._num_devices = num_devices
        self._num_stages = num_stages
        self._this_rank = rank
        self._micro_size = micro_size

        if num_devices <= 0 or num_stages <= 0:
            raise ValueError("The number of pipelines, devices, and stages should be positive integers")
        
        if rank < 0 or rank >= num_devices:
            raise ValueError("The rank of the current process should be in the range of [0, num_devices)")

        if num_devices % num_stages != 0:
            raise ValueError("The number of devices should be a multiple of the number of stages")
        
        if micro_size < num_stages:
            raise ValueError("The size of micro-batch should be no less than the number of stages")
        
        self._stage_mgr = IFIBPipelineRankStageManager(num_devices, num_stages, rank)

        self._construct()

    def _construct(self):
        self._schedule = []

        for rank in range(self._num_stages):
            sched = []
            q: deque[int] = deque()
            micro_counter: int = 0

            num_warmup_steps = self._num_stages - rank - 1
            num_idle_steps = rank

            def forward():
                nonlocal micro_counter
                micro_id = micro_counter
                micro_counter += 1
                q.append(micro_id)
                sched.append(ScheduleCell(CellType.FORWARD, micro_id, 0, rank))
            
            def backward():
                micro_id = q.popleft()
                sched.append(ScheduleCell(CellType.BACKWARD, micro_id, 0, rank))
            
            # warmup phase
            for _ in range(num_idle_steps):
                sched.append(ScheduleCell(CellType.IDLE))

            for _ in range(num_warmup_steps):
                forward()
            
            # 1f1b phase
            for _ in range(self._micro_size - num_warmup_steps):
                forward()
                backward()
            
            for _ in range(num_warmup_steps):
                backward()
            
            sched.append(ScheduleCell(CellType.SYNC, stage_id=rank))

            self._schedule.append(sched)


    def get_schedule(self, rank: int) -> List[ScheduleCell]:
        """
        Get the pipeline schedule.

        Returns:
            List[ScheduleCell]: The pipeline schedule.
        """

        if rank < 0 or rank >= self._num_devices:
            raise ValueError("The rank of the current process should be in the range of [0, num_devices)")
        
        device_group_size = self._num_devices // self._num_stages
        group_rank = rank // device_group_size

        return self._schedule[group_rank]
    
    def __str__(self) -> str:
        result = 'IFIBPipelineScheduleManager(\n'
        result += f'  num_stages = {self._num_stages},\n'
        result += f'  micro_size = {self._micro_size},\n'
        result += '  schedule = [\n'
        for i in range(self._num_stages):
            result += '    '
            for cell in self._schedule[i]:
                if cell.is_idle():
                    result += '    '
                else:
                    result += '{0: >3}{1}'.format(cell.micro_id, cell.type.value)
            result += '\n'
        result += '  ]\n'
        result += ')\n'

        return result
