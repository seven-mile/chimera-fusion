#!/bin/bash

source scripts/config.sh

base_dir=bert_prof
main_event_text=call_pipeline

if [ $pipeline == 'chimera' ]; then
    name=${model}_${chimera_pipelines}${pipeline}_${grad_reduce_method}_${stages}stages_${ngpus}gpus_microbs${microbs}_acc${acc}
elif [ $pipeline == 'interleaved' ]; then
    name=${model}_${interleaved_chunks}${pipeline}_${stages}stages_${ngpus}gpus_microbs${microbs}_acc${acc}
else
    name=${model}_${pipeline}_${stages}stages_${ngpus}gpus_microbs${microbs}_acc${acc}
fi

sqlite_paths=$(find ${base_dir} -type f -name "${name}_node*.sqlite" | sort )
job_ids=()

for sqlite_path in $sqlite_paths
do
    pickle_path_timeline=${base_dir}/$(basename ${sqlite_path} | cut -f 1 -d '.' )_timeline.pickle
    echo parse $sqlite_path
    python scripts/parse_nvtx_events.py \
        $sqlite_path \
        --pickle_path_timeline $pickle_path_timeline \
        --ignore_first_event \
        --main_event_indices '5,6,7' \
        --event_keywords call_forward,call_backward,cov_kron_A,cov_kron_B,inv_kron_A,inv_kron_B,precondition,reduce,gather,sync,optimizer \
        --main_event_text $main_event_text &
    job_ids+=($!)
done

echo waiting for jobs ${job_ids[@]} ...
for job_id in ${job_ids[@]}
do
    wait $job_id
done

echo cleaning up ...
for sqlite_path in $sqlite_paths
do
    rm -f $sqlite_path
    nsys_path=${base_dir}/$(basename ${sqlite_path} | cut -f 1 -d '.' ).nsys-rep
    rm -f $nsys_path
done

pickle_paths=""
for pickle_path in $(find ${base_dir} -type f -name "${name}_node*_timeline.pickle" | sort )
do
    pickle_paths+="${pickle_path},"
done
fig_path=${base_dir}/${name}.pdf
echo creating ${fig_path} ...
echo "${name}_node*_timeline.pickle"
python scripts/plot_cuda_timeline.py \
    $pickle_paths \
    --fig_path $fig_path \
    --title $name \
    --num_replicas 1 \
    >> plot_cuda_time.txt
#imgcat $fig_path
