import copy
from dataclasses import dataclass
from enum import Enum
from typing import List


class PipelineRankStageManager:
    """
    PipelineRankStageManager is an abstract class that manages the mapping between
    pipeline stages and device ranks.
    """

    def get_rank_to_stage_map(self, pipeline_id: int) -> List[int]:
        """
        Get the rank to stage map of the pipeline with the given index.

        Args:
            pipeline_id (int): The index of the pipeline.

        Returns:
            List[int]: The rank to stage map.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def get_stage_to_rank_map(self, pipeline_id: int) -> List[int]:
        """
        Get the stage to rank map of the pipeline with the given index.

        Args:
            pipeline_id (int): The index of the pipeline.

        Returns:
            List[int]: The stage to rank map.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError


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

    def get_rank_to_stage_map(self, pipeline_id: int) -> List[int]:
        """
        Get the rank to stage map of the pipeline with the given index.

        Args:
            pipeline_id (int): The index of the pipeline.

        Returns:
            List[int]: The rank to stage map.
        """
        assert pipeline_id == 0, "DAPPLE only supports one pipeline"
        return self._rank_to_stage_map
    
    def get_stage_to_rank_map(self, pipeline_id: int) -> List[int]:
        """
        Get the stage to rank map of the pipeline with the given index.

        Args:
            pipeline_id (int): The index of the pipeline.

        Returns:
            List[int]: The stage to rank map.
        """
        assert pipeline_id == 0, "DAPPLE only supports one pipeline"
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


class AutoGeneratePipelineRank:

    def __init__(self, stage_numbers, divisors, micro_batch_numbers):
        """
        初始化数据
        stage_numbers为stage的数量
        divisors则为Q的因子。也就是上行下行流水线的总数量
        将stage_numbers // (divisors)作为每条pipeline的micro-batch的数量。
        micro_batch_numbers 为微批的数量
        """
        self.module_to_stage_map = [i for i in range(stage_numbers)]
        self.stage_numbers = stage_numbers
        # up和down流水线个数
        assert divisors % 2 == 0, "pipeline 必须是偶数个"
        self.pipeline_numbers = 1 if divisors is None else divisors//2
        # 总微批的数量
        self.micro_batch_numbers = micro_batch_numbers
        self.push_pipeline_numbers = {
            "up": 0,
            "down": 0
        }
        self.push_micro_batch = 0

    def generate_pipeline(self):
        # 构建pipeline
        self.up_pipline_list = []
        self.down_pipeline_list = []
        for i in range(self.pipeline_numbers):
            # 构建up pipeline
            micro_num = self.stage_numbers//(2*self.pipeline_numbers)
            if self.micro_batch_numbers-self.push_micro_batch <= micro_num:
                micro_num = self.micro_batch_numbers-self.push_micro_batch

            self.push_micro_batch += micro_num

            self.up_pipline_list.append(MyPipeLine(i,
                                                   micro_num, self.stage_numbers,
                                                   self.pipeline_numbers, self.module_to_stage_map, True))

            if self.micro_batch_numbers-self.push_micro_batch <= micro_num:
                micro_num = self.micro_batch_numbers-self.push_micro_batch

            self.push_micro_batch += micro_num

            # 构建down pipeline
            self.down_pipeline_list.append(MyPipeLine(i,
                                                      micro_num, self.stage_numbers,
                                                      self.pipeline_numbers, self.module_to_stage_map, False))

    def get_schedule(self, is_iteration=False):
        schedule = []
        schedule_up_down = []
        pipelines = self.up_pipline_list + self.down_pipeline_list

        for i in range(self.stage_numbers):
            schedule.append(list())
            schedule_up_down.append(list())
            # 生成stage
        # 判断流水线是否全部结束
        has_next_flag = True
        has_next_sync = 0
        # 当前时间阶段
        steps = 0
        # 用于确定同步梯度
        sync_list = [[] for i in range(self.stage_numbers)]
        while(has_next_flag or has_next_sync != 0):
            next_flag = False
            # 迭代方式时返回
            sub_schedule = list("" for _ in range(self.stage_numbers))
            for index, pipeline in enumerate(pipelines):
                if pipeline.has_next_pass():
                    # 如果有下一阶段
                    next_data, is_pop, step_direction, up_or_down, is_sync = pipeline.next_pass()

                    for k in next_data.keys():
                        schedule[next_data[k] %
                                 self.stage_numbers].append(str(k))
                        up_or_down_str = str(index)
                        up_or_down_str += "@down@" if up_or_down else "@up@"
                        schedule_up_down[next_data[k] %
                                         self.stage_numbers].append(f"{up_or_down_str}{'f' if step_direction[k] == 1 else 'b'}")
                        if step_direction.get(pipeline.micro_batch_ids[-1], 1) != 1:
                            direction = "down"
                            if pipeline.up_or_down:
                                direction = "up"

                        sub_schedule[next_data[k] %
                                     self.stage_numbers] = f"{up_or_down_str}{'f' if step_direction[k] == 1 else 'b'}"
                    # 计算同步梯度。
                    if is_sync and next_data.get(pipeline.micro_batch_ids[-1]) is not None:
                        has_next_sync += 1
                        sync_list[next_data[pipeline.micro_batch_ids[-1]] %
                                  self.stage_numbers].append(f"{up_or_down_str}s")
                    if is_pop and pipeline.has_next_pass():
                        # 当前如果有一个micro_batch结束，则添加新的进队列
                        micro_num = self.stage_numbers//(2 *
                                                         self.pipeline_numbers)
                        if self.micro_batch_numbers-self.push_micro_batch <= micro_num:
                            micro_num = self.micro_batch_numbers-self.push_micro_batch

                        self.push_micro_batch += micro_num
                        if micro_num != 0:
                            if pipeline.up_or_down:
                                direction = "up"
                            else:
                                direction = "down"
                            pipelines.append(MyPipeLine(self.pipeline_numbers+self.push_pipeline_numbers[direction],
                                                        micro_num, self.stage_numbers,
                                                        self.pipeline_numbers, self.module_to_stage_map, pipeline.up_or_down))
                            self.push_pipeline_numbers[direction] += 1

                    next_flag = True
            for index, s in enumerate(sub_schedule):
                if s == "" and len(sync_list[index]) > 0:
                    sub_schedule[index] = sync_list[index].pop(0)
                    has_next_sync -= 1

            for i in range(self.stage_numbers):
                if len(schedule[i]) <= steps:
                    schedule[i].append(sub_schedule[i])
                    schedule_up_down[i].append(sub_schedule[i])

            steps += 1
            has_next_flag = next_flag
            if is_iteration and has_next_flag:
                # 返回当前时刻各个stage的状态 包括
                # 空字符串， 不执行
                # forward/backward 前向还是后向
                # down/up  升向还是降向
                yield sub_schedule
        # print("==========================================")
        # for i in schedule:
        #     for j in i:
        #         print("%2s" % (j), end=" ")
        #     print("")
        # for index, i in enumerate(schedule_up_down):
        #     for j in i:
        #         print("%2s" % (j), end=" ")
        #     print("")


class MyPipeLine:
    def __init__(self, pipeline_id, micro_batch_numbers,
                 stage_numbers, pipeline_numbers, module_to_stage_map, up_or_down):

        self.pipeline_id = pipeline_id
        self.micro_batch_numbers = micro_batch_numbers
        self.stage_to_rank_map = None
        self.pipeline_numbers = pipeline_numbers
        self.stage_numbers = stage_numbers
        self.module_to_stage_map = module_to_stage_map
        self.up_or_down = up_or_down
        self.devices = None

        # 打印数据使用
        self.steps = -1
        self.step_direction = dict()
        self.micro_batch_ids = list()
        self.micro_batch_device = dict()
        micro_batch_id = ((self.pipeline_id//2)*self.stage_numbers)
        micro_batch_id += (0 if self.up_or_down else self.stage_numbers//2)
        for x in range(self.micro_batch_numbers):
            self.micro_batch_ids.append(
                x+(self.pipeline_id % self.pipeline_numbers)*(self.stage_numbers//self.pipeline_numbers//2)+micro_batch_id)

        # 生成映射表
        # if self.stage_to_rank_map is not None:
        #     return self.stage_to_rank_map
        # 计算出前向第一个stage对应设备的位置
        start_stage_device = (self.pipeline_id % self.pipeline_numbers) * \
            (self.stage_numbers // self.pipeline_numbers)
        self.devices = [x for x in self.module_to_stage_map[start_stage_device:] +
                        self.module_to_stage_map[:start_stage_device]]

        if self.up_or_down is True:
            # down pipeline
            self.stage_to_rank_map = {
                str(index): [device] for index, device in enumerate(self.devices)}
        else:
            # up pipeline
            self.stage_to_rank_map = {
                str(self.stage_numbers-1-index): [device] for index, device in enumerate(self.devices)}

    def next_pass(self):
        # 下一阶段对应的设备
        if self.steps <= (self.micro_batch_numbers-1) * 2:
            self.steps += 1

        over_back_micro_batch = []
        for micro_batch in self.micro_batch_device.keys():
            # 如果是降向pipeline，则每运行一次step+1，即对应的stage+1
            # 如果是升向pipeline，则与之相反。
            step = 1 if self.up_or_down else -1
            # 判断当前micro_batch所在的device与初始device之间的差距，如果走完一圈，则方向取反。
            if self.step_direction[micro_batch] == 1 and abs(self.micro_batch_device[micro_batch] - (self.stage_to_rank_map["0"][0] + 2*self.stage_numbers)) >= self.stage_numbers-1:
                self.step_direction[micro_batch] = -1
            elif self.step_direction[micro_batch] == -1 and self.micro_batch_device[micro_batch] == self.stage_to_rank_map["0"][0] + 2*self.stage_numbers:
                # 反向结束 = 加入到待删除列表
                over_back_micro_batch.append(micro_batch)
            else:
                self.micro_batch_device[micro_batch] += step * \
                    self.step_direction[micro_batch]
        pop_one = False
        for micro_batch in over_back_micro_batch:
            # 删除掉反向传播后的micro_batch
            self.micro_batch_device.pop(micro_batch)
            pop_one = True

        if self.steps % 2 == 0:
            # 每走两步，如果当前pipeline还有micro batch要处理，则添加进来。
            # 定位到stage0的设备位置
            self.micro_batch_device[self.micro_batch_ids[self.steps //
                                                         2]] = self.stage_to_rank_map["0"][0] + 2*self.stage_numbers
            self.step_direction[self.micro_batch_ids[self.steps // 2]] = 1
        is_sync = True if self.step_direction.get(
            self.micro_batch_ids[-1]) == -1 else False
        return self.micro_batch_device, pop_one, self.step_direction, self.up_or_down, is_sync

    def has_next_pass(self):
        """判断流水线是否结束"""
        if self.micro_batch_numbers > 0 and (self.steps == -1 or self.micro_batch_device):
            return True
        return False


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
