#!/bin/bash

#model=bert_base
model=bert_large
#model=gpt2_base
#model=gpt2_large
data_path=data/${model}_time_memory.csv

fig_path=figs/${model}_perf_model_gpipe_1f1b.pdf
python scripts/plot_perf_model.py \
    --data_path $data_path \
    --fig_path $fig_path \

imgcat $fig_path

fig_path=figs/${model}_perf_model_chimera.pdf
python scripts/plot_perf_model.py \
    --data_path $data_path \
    --fig_path $fig_path \
    --chimera \

imgcat $fig_path
