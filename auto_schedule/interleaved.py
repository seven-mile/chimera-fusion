from .proto import CellType, PipelineRankStageManager, PipelineScheduleManager, ScheduleCell

from typing import List

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


class InterleavedPipelineScheduleManager(PipelineScheduleManager):
    """
    PipelineScheduleManager for Interleaved 1F1B pipeline scheduling.

    The pipeline schedule is a list of time steps, where each step is a list of cells for the corresponding ranks.
    """
    def __init__(self, num_chunks: int, num_devices: int, num_stages: int, rank: int, micro_size: int):
        """
        Initialize the InterleavedPipelineScheduleManager.

        Args:
            num_chunks (int): The number of interleaved chunks.
            num_devices (int): The number of devices.
            num_stages (int): The number of stages.
            ranks (List[int]): The ranks of the devices.
        """

        self._num_chunks = num_chunks
        self._num_devices = num_devices
        self._num_stages = num_stages
        self._this_rank = rank
        self._micro_size = micro_size

        if num_devices <= 0 or num_stages <= 0:
            raise ValueError("The number of devices and stages should be positive integers")
        
        if rank < 0 or rank >= num_devices:
            raise ValueError("The rank of the current process should be in the range of [0, num_devices)")
        
        if num_stages % num_chunks != 0:
            raise ValueError("The number of stages should be a multiple of the number of chunks")

        if num_devices % self.num_stages_per_chunk != 0:
            raise ValueError("The number of devices should be a multiple of the number of stages per chunk")

        if micro_size < self.num_stages_per_chunk:
            raise ValueError("The micro size should be greater than or equal to the number of stages per chunk")
        
        self._stage_mgr = InterleavedPipelineRankStageManager(num_chunks, num_devices, num_stages, rank)

        self._construct()

    @property
    def num_stages_per_chunk(self):
        return self._num_stages // self._num_chunks

    def _construct(self):
        self._schedule: List[List[ScheduleCell]] = []

        device_group_size = self._num_devices // self.num_stages_per_chunk

        for rank in range(self.num_stages_per_chunk):
            chunk_to_stage = [
                self._stage_mgr.get_rank_to_stage_map(chunk_id)[rank * device_group_size]
                for chunk_id in range(self._num_chunks)
            ]

            sched = []
            forward_counter: int = 0
            backward_counter: int = 0

            # warm up in the last chunk
            num_warmup_steps = self.num_stages_per_chunk - rank - 1
            num_warmup_steps += (self._num_chunks - 1) * self.num_stages_per_chunk

            num_idle_steps = rank

            def calc_micro_id(counter: int):
                high_part = counter // self._num_stages
                low_part = counter % self.num_stages_per_chunk
                return high_part * self.num_stages_per_chunk + low_part

            def forward():
                nonlocal forward_counter

                micro_id = calc_micro_id(forward_counter)
                chunk_id = forward_counter // self.num_stages_per_chunk % self._num_chunks
                stage_id = chunk_to_stage[chunk_id]

                sched.append(ScheduleCell(CellType.FORWARD, micro_id, chunk_id, stage_id))

                forward_counter += 1
            
            def backward():
                nonlocal backward_counter

                micro_id = calc_micro_id(backward_counter)
                chunk_id = backward_counter // self.num_stages_per_chunk % self._num_chunks
                # backward should be reversed chunk order
                chunk_id = self._num_chunks - 1 - chunk_id
                stage_id = chunk_to_stage[chunk_id]

                sched.append(ScheduleCell(CellType.BACKWARD, micro_id, chunk_id, stage_id))

                backward_counter += 1
            
            # warmup phase
            for _ in range(num_idle_steps):
                sched.append(ScheduleCell(CellType.IDLE))

            for _ in range(num_warmup_steps):
                forward()
            
            # 1f1b phase
            for _ in range(self._micro_size * self._num_chunks - num_warmup_steps):
                forward()
                backward()
            
            for _ in range(num_warmup_steps):
                backward()
            
            for chunk_id in range(self._num_chunks):
                stage_id = chunk_to_stage[chunk_id]
                sched.append(ScheduleCell(CellType.SYNC, stage_id=stage_id))

            self._schedule.append(sched)


    def get_schedule(self, rank: int) -> List[ScheduleCell]:
        """
        Get the pipeline schedule.

        Returns:
            List[ScheduleCell]: The pipeline schedule.
        """

        if rank < 0 or rank >= self._num_devices:
            raise ValueError("The rank of the current process should be in the range of [0, num_devices)")
        
        device_group_size = self._num_devices // self.num_stages_per_chunk
        group_rank = rank // device_group_size

        return self._schedule[group_rank]
    
    def __str__(self) -> str:
        result = 'InterleavedPipelineScheduleManager(\n'
        result += f'  num_chunks = {self._num_chunks},\n'
        result += f'  num_stages = {self._num_stages},\n'
        result += f'  micro_size = {self._micro_size},\n'
        result += f'  num_stages_per_chunk = {self.num_stages_per_chunk},\n'
        result += '  schedule = [\n'
        for i in range(self.num_stages_per_chunk):
            result += '   '
            for cell in self._schedule[i]:
                if cell.is_idle():
                    result += '    '
                else:
                    result += '{0: >2}{1}{2}'.format(' ' if cell.micro_id < 0 else cell.micro_id, cell.type.value, cell.stage_id)
            result += '\n'
        result += '  ]\n'
        result += ')\n'

        return result
