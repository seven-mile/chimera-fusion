
from chimera_pipeline_rank import ChimeraScheduleManager, ChimeraPipelineRankStageManager, DapplePipelineRankStageManager


if __name__ == '__main__':

    num_pipelines = 2
    num_stages = 4
    micro_size = 4
    
    this_rank = 0
    num_devices = num_stages * 2

    chimera_prs = ChimeraPipelineRankStageManager(num_pipelines, num_devices, num_stages, this_rank)
    dapple_prs = DapplePipelineRankStageManager(num_devices, num_stages, this_rank)

    print(chimera_prs.get_rank_to_stage_map(0), chimera_prs.get_rank_to_stage_map(1))
    print(dapple_prs.get_rank_to_stage_map())
    print(dapple_prs.get_stage_to_rank_map())

    print(chimera_prs.get_stage_to_ranks_map())
    print(dapple_prs.get_stage_to_ranks_map())

    # Create a pipeline rank stage manager object
    pipeline_rank_stage_manager = ChimeraScheduleManager(num_pipelines, num_devices, num_stages, this_rank, micro_size)

    print(pipeline_rank_stage_manager)
