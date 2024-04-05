import copy
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class PipelineRankStageManager:
    """
    PipelineRankStageManager is an abstract class that manages the mapping between
    pipeline stages and device ranks.
    """

    @property
    def num_prs_keys(self):
        """
        Get the number of PRS keys.
        
        `pipeline_id` for Chimera, `chunk_id` for Interleaved, `1` for DAPPLE.

        Returns:
            int: The number of PRS keys.
        """
        raise NotImplementedError
    
    @property
    def num_stages(self):
        """
        Get the number of stages.

        Returns:
            int: The number of stages.
        """
        raise NotImplementedError
    
    @property
    def num_devices(self):
        """
        Get the number of devices.

        Returns:
            int: The number of devices.
        """
        raise NotImplementedError

    def get_rank_to_stage_map(self, key: Optional[int] = None) -> List[int]:
        """
        Get the rank to stage map of the pipeline with the given index.

        Args:
            key (int): The index of the mapping. \
                `pipeline_id` for Chimera, `chunk_id` for Interleaved, \
                `None` for DAPPLE.

        Returns:
            List[int]: The rank to stage map.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def get_stage_to_rank_map(self, key: Optional[int] = None) -> List[int]:
        """
        Get the stage to rank map of the pipeline with the given index.

        Args:
            key (int): The index of the mapping. \
                `pipeline_id` for Chimera, `chunk_id` for Interleaved, \
                `None` for DAPPLE.

        Returns:
            List[int]: The stage to rank map.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError
    
    def get_stage_to_ranks_map(self) -> List[List[int]]:
        """
        Get the stage to ranks map.

        Returns:
            List[List[int]]: The stage to ranks map.
        """
        stage_to_ranks = [set() for i in range(self.num_stages)]
        for _rank in range(self.num_devices):
            for _key in range(self.num_prs_keys):
                stage = self.get_rank_to_stage_map(_key)[_rank]
                stage_to_ranks[stage].add(_rank)
        
        return [list(ranks) for ranks in stage_to_ranks]


class ChimeraPipelineRankStageManager(PipelineRankStageManager):
    """
    PipelineRankStageManager for Chimera pipeline scheduling.

    The mapping is stored in two lists of lists:
    - `stage_to_rank_map[pipeline_id][stage_id]` is the rank of the device
    that runs the stage with the given index in the pipeline with the given index.
    - `rank_to_stage_map[pipeline_id][rank]` is the index of the stage
    that runs on the device with the given rank in the pipeline with the given index.

    All of "index", "id" in the following description is 0-based.

    Notice that you have to provide the rank of current process.

    This is because for pipeline parallelism, we may have multiple devices
    for one single stage, which makes inner parallelism like DP, SP, TP possible.

    Then we have to know the rank of the current process to determine which device
    in the corresponding ranks should be used.

    For example, if we have 4 devices and 2 stages, and need two chimera pipelines.

    From the viewpoint of rank 0, the stage to rank map may look like:
    ```
    [
        [0, 2],
        [2, 0]
    ]
    ```
    From the viewpoint of rank 1, the stage to rank map may look like:
    ```
    [
        [1, 3],
        [3, 1]
    ]
    ```
    """
    def __init__(self, num_pipelines: int, num_devices: int, num_stages: int, rank: int):
        """
        Initialize the ChimeraPipelineRankStageManager.

        Args:
            num_pipelines (int): The number of pipelines.
            num_devices (int): The number of devices.
            num_stages (int): The number of stages.
            rank (int): The rank of the current process.

        Raises:
            ValueError: If current rank or the number of pipelines, devices,
            or stages is invalid.
        """

        if num_pipelines <= 0 or num_devices <= 0 or num_stages <= 0:
            raise ValueError("The number of pipelines, devices, and stages should be positive integers")

        if rank < 0 or rank >= num_devices:
            raise ValueError("The rank of the current process should be in the range of [0, num_devices)")

        if num_devices % num_stages != 0:
            raise ValueError("The number of devices should be a multiple of the number of stages")
        
        if num_pipelines & (num_pipelines - 1) != 0:
            raise ValueError("The number of pipelines should be a power of 2")
        
        if num_pipelines > num_stages:
            raise ValueError("The number of pipelines should not be greater than the number of stages")

        self._num_pipelines = num_pipelines
        self._num_devices = num_devices
        self._num_stages = num_stages
        self._this_rank = rank
        self._construct()

    @property
    def num_prs_keys(self):
        return self._num_pipelines
    
    @property
    def num_stages(self):
        return self._num_stages
    
    @property
    def num_devices(self):
        return self._num_devices

    def get_rank_to_stage_map(self, pipeline_id: int) -> List[int]:
        """
        Get the rank to stage map of the pipeline with the given index.

        Args:
            pipeline_id (int): The index of the pipeline.

        Returns:
            List[int]: The rank to stage map.
        """
        return self._rank_to_stage_map[pipeline_id]

    def get_stage_to_rank_map(self, pipeline_id: int) -> List[int]:
        """
        Get the stage to rank map of the pipeline with the given index.

        Args:
            pipeline_id (int): The index of the pipeline.

        Returns:
            List[int]: The stage to rank map.
        """
        return self._stage_to_rank_map[pipeline_id]

    def _construct(self):
        self._stage_to_rank_map = [[-1 for _ in range(self._num_stages)] for _ in range(self._num_pipelines)]
        self._rank_to_stage_map = [[-1 for _ in range(self._num_devices)] for _ in range(self._num_pipelines)]
        
        per_stage_devices = self._num_devices // self._num_stages

        def _get_pipeline_meta(pipeline_id):
            """
            Get the start rank and step of the pipeline with the given index.
            """
            # Up or down pipelines has different start ranks
            threshold = self._num_pipelines // 2

            if pipeline_id < threshold:
                start_rank = pipeline_id * self._num_devices // threshold
                step = 1
            else:
                start_rank = (pipeline_id - threshold) * self._num_devices // threshold
                start_rank = self._num_devices - start_rank - per_stage_devices
                step = -1
            return start_rank, step

        for pipeline_id in range(self._num_pipelines):
            start_rank, step = _get_pipeline_meta(pipeline_id)

            for rank in range(self._num_devices):
                if rank % per_stage_devices == 0:
                    calc_rank = start_rank + self._this_rank % per_stage_devices
                    self._stage_to_rank_map[pipeline_id][rank // per_stage_devices] = calc_rank
                
                offset = 0
                if step == -1:
                    offset = per_stage_devices - 1
                self._rank_to_stage_map[pipeline_id][(start_rank + offset) % self._num_devices] = rank // per_stage_devices

                start_rank = (start_rank + step + self._num_devices) % self._num_devices


class DapplePipelineRankStageManager(PipelineRankStageManager):
    """
    PipelineRankStageManager for DAPPLE pipeline scheduling.

    The mapping is straightforward:
    - stage index is rank id
    - rank id is the corresponding rank among the devices of the pipeline stage
    """
    def __init__(self, num_devices: int, num_stages: int, rank: int):
        """
        Initialize the DapplePipelineRankStageManager.

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


class CellType(Enum):
    """
    An enumeration of the types of cells in the schedule table.
    """
    IDLE = 'i'
    FORWARD = 'f'
    BACKWARD = 'b'
    SYNC = 's'


@dataclass
class ScheduleCell:
    """
    A data class representing a cell in the schedule table.
    """
    type: CellType = CellType.IDLE
    micro_id: int = -1
    pipeline_id: int = -1
    stage_id: int = -1
    time_step: int = -1
    forward_double: bool = False

    def is_forward(self):
        return self.type == CellType.FORWARD

    def is_sync(self):
        return self.type == CellType.SYNC

    def is_idle(self):
        return self.type == CellType.IDLE


class PipelineScheduleManager:
    """
    PipelineScheduleManager is an abstract class that produces the schedule of the pipelines.
    """

    def get_schedule(self, rank: int) -> List[ScheduleCell]:
        """
        Get the schedule of the pipeline with the given rank.

        Args:
            rank (int): The rank of the pipeline.

        Returns:
            List[ScheduleCell]: The schedule, `schedule[time_step][rank]` is the cell in the schedule table.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError


class BlockType(Enum):
    """
    An enumeration of the types of blocks in the schedule table.
    """
    FORWARD = 'f'
    FORWARD_DOUBLE = 'd'
    BACKWARD = 'b'

    def to_cell_type(self):
        if self == BlockType.FORWARD:
            return CellType.FORWARD
        if self == BlockType.FORWARD_DOUBLE:
            return CellType.FORWARD
        if self == BlockType.BACKWARD:
            return CellType.BACKWARD
        raise ValueError(f"Invalid block type: {self}")


class ChimeraBlock:
    """
    A class representing a Chimera block in the Chimera pipeline scheduling.
    """
    def __init__(self,
                 block_type: BlockType,
                 num_pipelines: int,
                 num_devices: int,
                 num_stages: int,
                 rank: int,
                 micro_size: int,
                 micros: List[List[int]],
                 start_micro_id: int):
        """
        Initialize the ChimeraBlock.

        Args:
            type (BlockType): The type of the block. It can be FORWARD, FORWARD_DOUBLE, or BACKWARD.
            num_pipelines (int): The number of pipelines.
            num_devices (int): The number of devices.
            num_stages (int): The number of stages.
            rank (int): The rank of the current process.
            micro_size (int): The size of micro-batch.
            micros (List[List[int]]): The micros to be scheduled.
            start_micro_id (int): The start micro id of this block.
        """
        self._stage_mgr = ChimeraPipelineRankStageManager(num_pipelines, num_devices, num_stages, rank)
        
        self._type = block_type
        self._num_pipelines = num_pipelines
        self._num_devices = num_devices
        self._num_stages = num_stages
        self._this_rank = rank
        self._micro_size = micro_size
        
        self.schedule: List[List[ScheduleCell]] = []

        self._construct(micros, start_micro_id)

    def _construct(self, micros: List[List[int]], start_micro_id: int):
        micro_per_pipeline = self._num_stages // self._num_pipelines
        per_stage_devices = self._num_devices // self._num_stages

        stage_map = [[0 for _ in range(micro_per_pipeline)] for _ in range(self._num_pipelines)]
        for pipeline_id in range(self._num_pipelines):
            for micro_id in range(micro_per_pipeline):
                stage_map[pipeline_id][micro_id] = -2 * micro_id

        while True:
            micro_inserted = True
            sub_schedule = [ScheduleCell() for _ in range(self._num_stages)]
            sub_schedule_dup = [ScheduleCell() for _ in range(self._num_stages)]

            for pipeline_id in range(self._num_pipelines):
                for micro_id in range(micro_per_pipeline):
                    stage_id = stage_map[pipeline_id][micro_id]
                    if stage_id < 0 or stage_id >= self._num_stages:
                        stage_map[pipeline_id][micro_id] += 1
                        continue

                    micro_inserted = False
                    if self._type == BlockType.BACKWARD:
                        first_stage_pipeline = (pipeline_id + self._num_pipelines // 2) % self._num_pipelines
                        step = -1 if pipeline_id < self._num_pipelines // 2 else 1
                    else:
                        first_stage_pipeline = pipeline_id
                        step = 1 if pipeline_id < self._num_pipelines // 2 else -1

                    first_stage_rank = self._stage_mgr.get_stage_to_rank_map(first_stage_pipeline)[0]
                    # group_rank is a virtual rank that does not consider inner parallelism
                    group_rank = first_stage_rank // per_stage_devices + step * stage_id
                    group_rank = (group_rank + self._num_stages) % self._num_stages
                    
                    if self._type == BlockType.FORWARD_DOUBLE:
                        micro_index = micro_id * 2
                    else:
                        micro_index = micro_id
                    
                    cell_ref = sub_schedule[group_rank]
                    cell_ref.pipeline_id = pipeline_id
                    if self._type == BlockType.BACKWARD:
                        cell_ref.stage_id = self._num_stages - 1 - stage_id
                    else:
                        cell_ref.stage_id = stage_id
                    cell_ref.type = self._type.to_cell_type()
                    cell_ref.micro_id = micros[pipeline_id][micro_index + start_micro_id]
                    cell_ref.forward_double = self._type == BlockType.FORWARD_DOUBLE

                    if self._type == BlockType.FORWARD_DOUBLE:
                        sub_schedule_dup[group_rank] = copy.copy(sub_schedule[group_rank])
                        sub_schedule_dup[group_rank].micro_id = micros[pipeline_id][micro_index + start_micro_id + 1]

                    stage_map[pipeline_id][micro_id] += 1
            
            if micro_inserted:
                break
            self.schedule.append(sub_schedule)
            if self._type == BlockType.FORWARD_DOUBLE:
                self.schedule.append(sub_schedule_dup)
    
    def __str__(self):
        result = 'ChimeraBlock(\n'
        result += f'  num_pipelines = {self._num_pipelines},\n'
        result += f'  num_stages = {self._num_stages},\n'
        # the real micro_size of a block is num_stages
        result += f'  micro_size = {self._num_stages},\n'
        result += '  schedule = [\n'
        for i in range(self._num_stages):
            result += '    '
            for j in range(len(self.schedule)):
                cell = self.schedule[j][i]
                if cell.is_idle():
                    result += '    '
                else:
                    result += '{0: >3}{1}'.format(cell.micro_id, cell.type.value)
            result += '\n'
        result += '  ]\n'
        result += ')\n'

        return result


class ChimeraScheduleManager:
    """
    A class that produces the schedule of the Chimera pipelines.
    """
    def __init__(self,
                 num_pipelines: int,
                 num_devices: int,
                 num_stages: int,
                 rank: int,
                 micro_size: int):
        """
        Initialize the ChimeraScheduleManager.

        Args:
            num_pipelines (int): The number of pipelines.
            num_devices (int): The number of devices.
            num_stages (int): The number of stages.
            rank (int): The rank of the current process.
            micro_size (int): The size of micro-batch.
        """
        self._num_pipelines = num_pipelines
        self._num_devices = num_devices
        self._num_stages = num_stages
        self._this_rank = rank
        self._micro_size = micro_size

        if num_pipelines <= 0 or num_devices <= 0 or num_stages <= 0:
            raise ValueError("The number of pipelines, devices, and stages should be positive integers")
        
        if rank < 0 or rank >= num_devices:
            raise ValueError("The rank of the current process should be in the range of [0, num_devices)")

        if num_devices % num_stages != 0:
            raise ValueError("The number of devices should be a multiple of the number of stages")
        
        if num_pipelines & (num_pipelines - 1) != 0:
            raise ValueError("The number of pipelines should be a power of 2")
        
        if num_pipelines > num_stages:
            raise ValueError("The number of pipelines should not be greater than the number of stages")
        
        if micro_size <= 0:
            raise ValueError("The size of micro-batch should be a positive integer")
        
        if micro_size % num_pipelines != 0:
            raise ValueError("The size of micro-batch should be a multiple of the number of pipelines")
        
        if micro_size % num_stages != 0:
            raise ValueError("The size of micro-batch should be a multiple of the number of stages")

        self._stage_mgr = ChimeraPipelineRankStageManager(num_pipelines, num_devices, num_stages, rank)

        self._construct()

    def _merge_chimera_block(former_block: ChimeraBlock, latter_block: ChimeraBlock):
        """
        Merge two chimera blocks.

        Args:
            former_block (ChimeraBlock): The former chimera block.
            latter_block (ChimeraBlock): The latter chimera block.
        
        Returns:
            List[List[ScheduleCell]]: The merged schedule.
        """
        result = []
        former = former_block.schedule
        latter = latter_block.schedule

        # prepend the first half of the former block
        for i in range(len(former) // 2):
            result.append(former[i])

        # former index and latter index
        fid, lid = len(former) // 2, 0

        # merge with two pointers
        while fid < len(former):
            double_former = True
            double_latter = True
            merge_one_step = False
            
            for rank in range(latter_block._num_stages):
                if fid > len(former) - 2 or former[fid+1][rank].is_idle() != former[fid][rank].is_idle():
                    double_former = False
                
                if lid > len(latter) - 2 or latter[lid+1][rank].is_idle() != latter[lid][rank].is_idle():
                    double_latter = False
                
                if former[fid][rank].is_idle() and not latter[lid][rank].is_idle():
                    former[fid][rank] = latter[lid][rank]
                    merge_one_step = True
            
            result.append(former[fid])
            if double_former:
                result.append(former[fid+1])
            
            if merge_one_step and double_latter:
                result.append(latter[lid+1])
            
            if merge_one_step:
                lid += double_latter + 1

            fid += double_former + 1

        while lid < len(latter):
            result.append(latter[lid])
            lid += 1
        
        return result

    def _construct(self):
        """
        Construct the schedule.
        """
        cur_micro_id = 0
        blocks: List[ChimeraBlock] = []
        micro_per_pipeline = self._micro_size // self._num_pipelines
        micro_per_pipeline_block = self._num_stages // self._num_pipelines
        micros = [[i * micro_per_pipeline + j for j in range(micro_per_pipeline)] for i in range(self._num_pipelines)]

        def append_block(block_type: BlockType):
            blocks.append(ChimeraBlock(block_type, self._num_pipelines, self._num_devices, self._num_stages, self._this_rank, self._micro_size, micros, cur_micro_id))

        while cur_micro_id < micro_per_pipeline:
            if cur_micro_id + 2 * micro_per_pipeline_block <= micro_per_pipeline:
                append_block(BlockType.FORWARD_DOUBLE)
                append_block(BlockType.BACKWARD)
                cur_micro_id += 2 * micro_per_pipeline_block
            else:
                append_block(BlockType.FORWARD)
                append_block(BlockType.BACKWARD)
                cur_micro_id += micro_per_pipeline_block

        for i in range(len(blocks)-1, 0, -1):
            blocks[i-1].schedule = ChimeraScheduleManager._merge_chimera_block(blocks[i-1], blocks[i])
        
        self.schedule = blocks[0].schedule
    
    def get_schedule(self, rank: int) -> List[ScheduleCell]:
        """
        Get the schedule of the pipeline with the given rank.

        Args:
            rank (int): The rank of the pipeline.

        Returns:
            List[ScheduleCell]: The schedule, `schedule[time_step][rank]` is the cell in the schedule table.
        """
        if rank < 0 or rank >= self._num_devices:
            raise ValueError("The rank of the pipeline should be in the range of [0, num_devices)")

        per_pipeline_devices = self._num_devices // self._num_stages
        group_rank = rank // per_pipeline_devices
        
        sched = []
        for i in range(len(self.schedule)):
            sched.append(self.schedule[i][group_rank])
        
        return sched
    
    def __str__(self):
        result = 'ChimeraScheduleManager(\n'
        result += f'  num_pipelines = {self._num_pipelines},\n'
        result += f'  num_stages = {self._num_stages},\n'
        result += f'  micro_size = {self._micro_size},\n'
        result += '  schedule = [\n'
        for i in range(self._num_stages):
            result += '    '
            for j in range(len(self.schedule)):
                cell = self.schedule[j][i]
                if cell.is_idle():
                    result += '    '
                else:
                    result += '{0: >3}{1}'.format(cell.micro_id, cell.type.value)
            result += '\n'
        result += '  ]\n'
        result += ')\n'

        return result


if __name__ == "__main__":
    num_stages = 8
    num_pipelines = 4
    num_devices = 16
    micro_size = 24
    this_rank = 0
    print(f"stage:{num_stages}  num_pipelines:{num_pipelines} micro_size:{micro_size}")

    stage_mgr = ChimeraPipelineRankStageManager(num_pipelines, num_devices, num_stages, this_rank)
    print('Stage To Rank')
    print(stage_mgr._stage_to_rank_map)
    print()
    print('Rank To Stage')
    print(stage_mgr._rank_to_stage_map)

    sched_mgr = ChimeraScheduleManager(num_pipelines, num_devices, num_stages, this_rank, micro_size)
    sched = sched_mgr.get_schedule(this_rank)
    print()
    print('Schedule')
    print(sched_mgr)

    exit(0)
