model=bert-large

# pipeline='gpipe'
# pipeline='1f1b'
pipeline='chimera'
# pipeline='interleaved'

stages=4
ngpus=4
microbs=32
acc=1

chimera_pipelines=2
interleaved_chunks=2

# grad_reduce_method='baseline'
# grad_reduce_method='stage'
grad_reduce_method='layer'
