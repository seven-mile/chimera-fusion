#!/bin/bash -l
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=output/prof_steps.out
#SBATCH --comment bupthpc

module load nvidia/cuda/11.8
conda activate pipefisher

export MASTER_ADDR=$(hostname)

source scripts/config.sh

export NSYS_NODE_INTERVAL=$((ngpus/stages))

if [ $pipeline == 'chimera' ]; then
    export NSYS_OUTPUT=bert_prof/${model}_${chimera_pipelines}${pipeline}_${stages}stages_${ngpus}gpus_microbs${microbs}_acc${acc}
elif [ $pipeline == 'interleaved' ]; then
    export NSYS_OUTPUT=bert_prof/${model}_${interleaved_chunks}${pipeline}_${stages}stages_${ngpus}gpus_microbs${microbs}_acc${acc}
else
    export NSYS_OUTPUT=bert_prof/${model}_${pipeline}_${stages}stages_${ngpus}gpus_microbs${microbs}_acc${acc}
fi

# Debug NCCL
# export NCCL_DEBUG=info
# export NCCL_DEBUG_SUBSYS=ALL

echo "Running with $pipeline pipeline, $stages stages, $ngpus gpus, $microbs microbatches, $acc gradient accumulation steps"

srun --wait=0 bash scripts/nsys_wrap.sh \
    python main_bert.py \
            --num_stages $stages \
            --corpus_path ./bert_data/wikipedia.segmented.nltk.txt \
            --vocab_path ./bert_data/bert-large-uncased-vocab.txt \
            --corpus_lines 10000 \
            --do_lower_case \
            --bert_config_path ./configs/bert_config_${model}-uncased.json \
            --max_seq_length 128 \
            --micro_batch_size $microbs \
            --num_optimization_steps 8 \
            --gradient_accumulation_steps $acc \
            --pipeline_method $pipeline \
            --p2p_backend 'gloo' \
            --collective_backend 'nccl' \
            --profile \
            --chunks $interleaved_chunks \
            --num_pipelines $chimera_pipelines
