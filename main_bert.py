import argparse
from itertools import chain
import os
import random
import math
import traceback
import yaml

import numpy as np
import torch
from torch import nn
from torch.cuda import nvtx
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from transformers import BertTokenizer, BertConfig
from model import BertAdam, BertDataset, get_stage_bert_for_pretraining

from pipeline import PipelineContext, PipelineExecutor, PipelineStage

try:
    import wandb
except ImportError:
    wandb = None


parser = argparse.ArgumentParser()
# Dataset & BERT
parser.add_argument("--corpus_path", default=None, type=str, required=True,
                    help="The input train corpus.")
parser.add_argument('--corpus_lines', default=None, type=int)
parser.add_argument("--vocab_path", type=str, required=True)
parser.add_argument("--on_memory", action='store_true',
                    help="Whether to load train samples into memory or use disk")
parser.add_argument("--do_lower_case", action='store_true',
                    help="Whether to lower case the input text. True for uncased models, False for cased models.")
parser.add_argument("--bert_config_path", type=str, required=True,
                    help="config to use.")
parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
# Training
parser.add_argument("--micro_batch_size", default=32, type=int,
                    help="Micro-batch size for training.")
parser.add_argument('--num_optimization_steps', default=None, type=int,
                    help="Total number of optimization steps to perform.")
parser.add_argument("--num_epochs", default=None, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--adam_learning_rate", default=3e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--adam_max_grad_norm", type=float, default=1.)
parser.add_argument("--beta1", default=0.9, type=float,
                    help="beta1 for Adam.")
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--warmup_proportion", default=0.1, type=float,
                    help="Proportion of training to perform linear learning rate warmup for.")
parser.add_argument("--damping", type=float, default=0.01)
# Pipeline
parser.add_argument('--pipeline_method', choices=[
                    '1f1b', 'gpipe', 'chimera', 'interleaved'], default='1f1b')
parser.add_argument("--chunks", default=2, type=int,
                    help="Number of chunks for interleaved 1f1b.")
parser.add_argument('--recompute', action='store_true',
                    help='Recompute activations in backward pass')
parser.add_argument('--num_stages', type=int, default=4,
                    help='number of stages in configurable BERT model')
parser.add_argument('--num_pipelines', type=int, default=2,
                    help='number of pipeline')
parser.add_argument('--layer_allreduce', action='store_true', help='whether to allreduce layer-wise')
# Others
parser.add_argument('--checkpoint_dir', default=None, type=str,
                    help='path to directory to save checkpoints')
parser.add_argument('--save_checkpoint_steps', type=int, default=200)
parser.add_argument('--seed', type=int, default=1,
                    help="random seed for initialization")
parser.add_argument('--p2p_backend', default=dist.Backend.GLOO, type=str)
parser.add_argument('--collective_backend', default=dist.Backend.NCCL, type=str)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--profile', action='store_true')

parser.add_argument('--observe_norm', action='store_true')
parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--wandb', action='store_true')


def main():
    total_steps = 0

    for epoch in range(num_epochs):
        dist.barrier()
        if pctx.num_replicas > 1:
            # deterministically shuffle based on epoch
            for train_loader in train_loaders:
                train_loader.sampler.set_epoch(epoch)
        
        steps_for_this_epoch = min(num_steps - total_steps, max_steps_per_epoch)

        train_one_epoch(epoch, total_steps, steps_for_this_epoch)
        total_steps += steps_for_this_epoch

    if pctx.is_master:
        print('Finished.')


def train_one_epoch(epoch, step, num_steps_for_this_epoch):

    num_p2p_comm = num_steps_for_this_epoch * pctx.num_micro_batches_per_step

    data_iters = list(map(iter, train_loaders))

    stage_dict = {stage.stage_id: stage for stage in stages}
    executor = PipelineExecutor(pctx, stage_dict, data_iters, num_p2p_comm)

    for i in range(num_steps_for_this_epoch):
        for optimizer in optimizers:
            optimizer.zero_grad()
        dist.barrier()
        
        nvtx.range_push('call_pipeline')

        loss = executor.run(step+i)

        nvtx.range_push('optimizer_step')
        for optimizer in optimizers:
            optimizer.step()
        nvtx.range_pop()
        
        nvtx.range_pop()

        if (step+i) % args.log_interval == 0:
            loss = torch.tensor(loss, device=pctx.device)
            dist.reduce(loss, dst=0)
            loss /= total_num_micro_batches_per_step
            if pctx.is_chimera:
                loss *= 2
            if pctx.is_master:
                print(f'epoch{epoch+1} step{step+i+1} loss = {float(loss)}')
                if args.wandb:
                    log = {'epoch': epoch+1, 'step': step+i+1, 'loss': float(loss),
                            'adam_learning_rate': optimizers[0].get_lr()[0]}
                    if args.observe_norm:
                        all_params = chain(*[stage.stage_module.parameters() for stage in stages])
                        log['p_norm'] = np.sqrt(sum([float(p.data.norm()) ** 2 for p in all_params]))
                        log['g_norm'] = np.sqrt(sum([float(p.grad.norm()) ** 2 for p in all_params]))
                    wandb.log(log)

        if args.checkpoint_dir is not None and (step+i+1) % args.save_checkpoint_steps == 0 and is_stage_master:
            state = {
                'epoch': epoch + 1,
                'model': [stage.stage_module.state_dict() for stage in stages],
                'optimizer': [optimizer.state_dict() for optimizer in optimizers]
            }
            assert os.path.isdir(args.checkpoint_dir)
            ckpt_file_path = os.path.join(args.checkpoint_dir, f'epoch{epoch+1}_step{step+i+1}_stage{"_".join(map(lambda s: str(s.stage_id), stages))}.pt')
            torch.save(state, ckpt_file_path)
            print(f'Saved checkpoint to {ckpt_file_path}')



if __name__ == "__main__":
    args = parser.parse_args()
    dict_args = vars(args)
    if args.config is not None:
        dict_args.update(yaml.safe_load(open(args.config, 'r')))
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    pctx = PipelineContext(
        p2p_backend=args.p2p_backend,
        collective_backend=args.collective_backend,
        pipeline_method=args.pipeline_method,
        num_stages=args.num_stages,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_pipelines=args.num_pipelines,
        num_chunks=args.chunks,
        recompute=args.recompute,
        max_seq_length=args.max_seq_length,
        is_layer_allreduce=args.layer_allreduce
    )

    print('got sched:', pctx.sched_mgr, flush=True)

    # Prepare BERT pipeline stages
    bert_config = BertConfig.from_json_file(args.bert_config_path)
    micro_batch_size = args.micro_batch_size
    max_seq_length = args.max_seq_length

    def get_pipeline_stage(prs_key):
        stage_id = pctx.stage_rank_mgr.get_rank_to_stage_map(prs_key)[pctx.world_rank]
        stage_module = get_stage_bert_for_pretraining(stage_id,
                                                      pctx.num_stages,
                                                      bert_config
                                                     ).to(pctx.device)

        return PipelineStage(pctx, stage_id, prs_key, stage_module)

    stages = [get_pipeline_stage(prs_key) for prs_key in range(pctx.num_prs_keys)]

    is_stage_master = pctx.world_rank % pctx.num_ranks_per_stage == 0

    # Prepare BERT dataset
    tokenizer = BertTokenizer(args.vocab_path, do_lower_case=args.do_lower_case)
    train_dataset = BertDataset(args.corpus_path,
                                tokenizer,
                                seq_len=max_seq_length,
                                corpus_lines=args.corpus_lines,
                                encoding='latin-1',
                                on_memory=args.on_memory)

    def get_train_loader(prs_key: int):
        sampler = None
        if pctx.num_replicas > 1:
            rank_in_replicas = rank_in_stage = pctx.world_rank % pctx.num_ranks_per_stage
            if pctx.is_chimera:
                rank_in_replicas = pctx.num_prs_keys * rank_in_stage + prs_key
            sampler = DistributedSampler(train_dataset, num_replicas=pctx.num_replicas, rank=rank_in_replicas)
        return DataLoader(train_dataset,
                          sampler=sampler,
                          batch_size=micro_batch_size,
                          drop_last=True,
                          num_workers=args.num_workers)

    train_loaders = [get_train_loader(prs_key) for prs_key in range(pctx.num_prs_keys)]

    # Set the number of optimization steps and epochs
    total_num_micro_batches_per_step = pctx.num_replicas * pctx.num_micro_batches_per_step
    total_num_samples_per_step = total_num_micro_batches_per_step * micro_batch_size
    max_steps_per_epoch = len(train_dataset) // total_num_samples_per_step
    num_steps = args.num_optimization_steps
    if num_steps is None:
        assert args.num_epochs, 'num_optimization_steps or num_epochs needs to be specified.'
        num_epochs = args.num_epochs
        num_steps = max_steps_per_epoch * args.num_epochs
    else:
        total_num_samples = num_steps * total_num_samples_per_step
        num_epochs = math.ceil(total_num_samples / len(train_dataset))

    # Prepare natural gradient preconditioners

    # Prepare optimizers
    def get_optimizer(module):
        decay_param_group = {'params': [], 'weight_decay': args.weight_decay}
        no_decay_param_group = {'params': [], 'weight_decay': 0.}
        for m in module.modules():
            if isinstance(m, nn.LayerNorm):
                no_decay_param_group['params'] += list(m.parameters())
            elif isinstance(m, (nn.Linear, nn.Embedding)):
                if hasattr(m, 'bias') and m.bias is not None:

                    no_decay_param_group['params'].append(m.bias)
                decay_param_group['params'].append(m.weight)
        params = [decay_param_group, no_decay_param_group]

        return BertAdam(params,
                        lr=args.adam_learning_rate,
                        b1=args.beta1,
                        warmup=args.warmup_proportion,
                        t_total=num_steps,
                        max_grad_norm=args.adam_max_grad_norm)

    optimizers = [get_optimizer(stage.stage_module) for stage in stages]

    dist.barrier()
    if pctx.is_master:
        if args.wandb:
            wandb.init(entity=os.getenv('WANDB_ENTITY'),
                       project=os.getenv('WANDB_PROJECT'))
            wandb.config.update(dict_args)
        print('============================')
        print(f'pipeline_method: {args.pipeline_method}')
        print(f'num_epochs: {num_epochs}')
        print(f'num_optimization_steps: {num_steps}')
        print(f'world_size: {pctx.world_size}')
        print(f'num_replica: {pctx.num_replicas}')
        print(f'num_pipeline: {pctx.num_pipelines}')
        print(f'num_micro_batches_per_step: {pctx.num_micro_batches_per_step}')
        print(f'recompute: {pctx.recompute}')
        print('----------------------------')
        for key, value in dict_args.items():
            print(f'{key}: {value}')
        print('============================')

    try:
        if args.profile:
            # with torch.cuda.profiler.profile():
            torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStart())
            main()
            torch.cuda.check_error(torch.cuda.cudart().cudaProfilerStop())
        else:
            main()
    except Exception as e:
        print('Exception in main thread:', type(e), e, flush=True)
        print(traceback.format_exc())
        raise
