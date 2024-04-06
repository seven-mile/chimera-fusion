model=bert-large

# pipeline='gpipe'
pipeline='1f1b'
# pipeline='chimera'
# pipeline='interleaved'

stages=8
ngpus=8
microbs=32
acc=1

chimera_pipelines=4
interleaved_chunks=2
