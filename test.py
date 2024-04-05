
from chimera_pipeline_rank import AutoGeneratePipelineRank, ChimeraScheduleManager, ChimeraPipelineRankStageManager


if __name__ == '__main__':

    num_pipelines = 2
    num_stages = 4
    micro_size = 4
    
    this_rank = 0
    num_devices = num_stages

    # Create a pipeline rank object
    autogen = AutoGeneratePipelineRank(num_stages, num_pipelines, micro_size)
    autogen.generate_pipeline()

    # Create a pipeline rank stage manager object
    pipeline_rank_stage_manager = ChimeraScheduleManager(num_pipelines, num_devices, num_stages, this_rank, micro_size)

    autogen_res = list(autogen.get_schedule(True))
    print(autogen_res)
    for i in range(num_stages):
        for j in range(len(autogen_res)):
            cell = autogen_res[j][i]
            if cell == '':
                print('    ', end='')
            else:
                micro, pipeline, ty = cell.split('@')
                micro = int(micro)
                print('{0: >3}{1}'.format(micro, ty), end='')
        print()

    print(pipeline_rank_stage_manager)

    

