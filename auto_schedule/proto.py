
from enum import Enum
from dataclasses import dataclass

from typing import List, Optional


class ScheduleMethod(Enum):
    GPIPE = 'gpipe'
    IFIB = '1f1b'
    INTERLEAVED = 'interleaved'
    CHIMERA = 'chimera'


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
    prs_key: int = 0
    stage_id: int = -1
    forward_double: bool = False

    def is_forward(self):
        return self.type == CellType.FORWARD

    def is_sync(self):
        return self.type == CellType.SYNC

    def is_idle(self):
        return self.type == CellType.IDLE


class PipelineRankStageManager:
    """
    PipelineRankStageManager is an abstract class that manages the mapping between
    pipeline stages and device ranks.
    """

    @property
    def num_prs_keys(self):
        """
        Get the number of PRS keys.
        
        `pipeline_id` for Chimera, `chunk_id` for Interleaved, `1` for 1F1B.

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
                `None` for 1F1B.

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
                `None` for 1F1B.

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
            List[ScheduleCell]: The schedule for this rank.

        Raises:
            NotImplementedError: This method should be implemented by subclasses.
        """
        raise NotImplementedError
