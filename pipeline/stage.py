import collections
from collections import deque
from typing import List, Tuple, Deque, OrderedDict, Iterator, Union, Dict
from contextlib import nullcontext

import torch
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.distributed as dist
from torch.cuda import nvtx

from .context import PipelineContext

import comm
from model import StageModule

class PipelineStage:

    def __init__(self, pctx: PipelineContext, stage_id: int, prs_key: int, stage_module: Union[StageModule, DistributedDataParallel]):
        self.pctx = pctx
        self.stage_id = stage_id
        self.prs_key = prs_key
        self.stage_module = stage_module

        self.input_output_queue: Deque[Tuple[OrderedDict[str, Tensor], OrderedDict[str, Tensor]]] = deque()

        self.total_loss = 0.
        self.nvtx_tag = ':up_pipe' if self.is_up_pipe else ''

        self.forward_recv_queues = {}
        self.backward_recv_queues = {}
        self.forward_send_queues = {}
        self.backward_send_queues = {}

        self.handles = []
        self.grads = []
        self.packed_grads = []

        self._init_comm_queues()
    
    @property
    def is_first_stage(self):
        return self.stage_id == 0

    @property
    def is_last_stage(self):
        return self.stage_id == self.pctx.num_stages - 1

    @property
    def prev_rank(self):
        if self.stage_id > 0:
            return self.pctx.stage_rank_mgr.get_stage_to_rank_map(self.prs_key)[self.stage_id - 1]
        else:
            return None
        
    @property
    def next_rank(self):
        if self.stage_id < self.pctx.num_stages - 1:
            return self.pctx.stage_rank_mgr.get_stage_to_rank_map(self.prs_key)[self.stage_id + 1]
        else:
            return None
    
    @property
    def is_up_pipe(self):
        return self.pctx.is_chimera and self.prs_key >= self.pctx.num_prs_keys // 2

    @property
    def p2p_tag(self):
        return self.prs_key

    @property
    def keys_from_source(self):
        if isinstance(self.stage_module, DistributedDataParallel):
            return self.stage_module.module.keys_from_source
        return self.stage_module.keys_from_source

    @property
    def sizes_from_prev_stage(self) -> Dict[str, Tuple]:
        stage_module = self.stage_module
        if isinstance(stage_module, DistributedDataParallel):
            stage_module = stage_module.module
        return stage_module.sizes_from_prev_stage

    @property
    def keys_from_prev_stage(self) -> List[str]:
        return list(self.sizes_from_prev_stage.keys())

    @property
    def sizes_for_next_stage(self) -> Dict[str, Tuple]:
        stage_module = self.stage_module
        if isinstance(stage_module, DistributedDataParallel):
            stage_module = stage_module.module
        return stage_module.sizes_for_next_stage

    @property
    def keys_for_next_stage(self):
        return list(self.sizes_for_next_stage.keys())
    
    @property
    def grad_sync_group(self):
        return self.pctx.sync_groups[self.stage_id]

    @property
    def is_distributed(self):
        return self.grad_sync_group is not None and self.grad_sync_group.size() > 1

    def send_outputs_to_queue(self, key, tensor):
        self.forward_send_queues[key].add(tensor)

    def send_input_grads_to_queue(self, key, tensor):
        self.backward_send_queues[key].add(tensor)

    def recv_inputs_from_queue(self, key):
        return self.forward_recv_queues[key].remove()

    def recv_output_grads_from_queue(self, key):
        return self.backward_recv_queues[key].remove()

    def call_forward(self, input_source: OrderedDict[str, Tensor]):
        nvtx.range_push('call_forward' + self.nvtx_tag)

        inputs = collections.OrderedDict()
        if not self.is_first_stage:
            for key in self.keys_from_prev_stage:
                inputs[key] = self.recv_inputs_from_queue(key)
        for key in self.keys_from_source:
            inputs[key] = input_source[key].to(self.pctx.device)
        assert len(inputs) > 0, 'No input is set.'

        no_grad_if_recompute = torch.no_grad if self.pctx.recompute else nullcontext
        with no_grad_if_recompute():
            try:
                outputs = self.stage_module(**inputs)
            except Exception as e:
                print(f'Error in stage{self.stage_id} calculation: {e}', flush=True)
                raise

        if not self.is_last_stage:
            for key in outputs:
                self.send_outputs_to_queue(key, outputs[key])
        else:
            self.total_loss += float(outputs['loss'])

        # push inputs/outputs to the queue
        self.input_output_queue.append((inputs, outputs))

        nvtx.range_pop()

    def call_backward(self, no_sync=True):
        nvtx.range_push('call_backward' + self.nvtx_tag)
        assert len(self.input_output_queue) > 0, 'No input/output is set.'
        # pop inputs/outputs from the queue
        inputs, outputs = self.input_output_queue.popleft()
        if self.pctx.recompute:
            with nvtx.range('recompute'):
                outputs = self.stage_module(**inputs)

        out_tensors = tuple(outputs.values())
        grad_tensors = None
        if not self.is_last_stage:
            grad_tensors = tuple(self.recv_output_grads_from_queue(key) for key in outputs)

        input_grads = collections.OrderedDict()

        def get_hook(key):
            def hook(grad):
                input_grads[key] = grad
            return hook

        if not self.is_first_stage:
            for key in self.keys_from_prev_stage:
                inputs[key].register_hook(get_hook(key))

        with self.no_sync_if_need(no_sync):
            torch.autograd.backward(out_tensors, grad_tensors=grad_tensors)
        if not self.is_first_stage:
            for key in self.keys_from_prev_stage:
                self.send_input_grads_to_queue(key, input_grads[key])

        del inputs, outputs

        nvtx.range_pop()

    def no_sync_if_need(self, no_sync: bool):
        if isinstance(self.stage_module, DistributedDataParallel) and no_sync:
            return self.stage_module.no_sync()
        return nullcontext()

    def _sync_grad_for_params_issue(self, param_grads: List[Tensor]):
        nvtx.range_push('sync_grad_layer' + self.nvtx_tag)

        assert self.grad_sync_group is not None, 'grad_sync_group is not specified.'

        packed_tensor = parameters_to_vector(param_grads)
        handle = dist.all_reduce(packed_tensor, group=self.grad_sync_group, async_op=True)
        self.handles.append(handle)
        self.packed_grads.append(packed_tensor)
        self.grads.append(param_grads)

        nvtx.range_pop()

    def install_sync_hooks(self):
        stage_module = self.stage_module.module if isinstance(self.stage_module, DistributedDataParallel) else self.stage_module

        def get_hook(layer_id):

            def sync_grad_hook(grad):
                print(f'Z layer {layer_id} sync_grad_hook', flush=True)
                self._sync_grad_for_params_issue([grad])
            
            return sync_grad_hook
        
        handles = []

        for idx, layer in enumerate(stage_module.layers):
            print(f'Z layer {idx} registering backward hook', flush=True)
            hook = get_hook(idx)
            for param in layer.parameters():
                if param.requires_grad:
                    handles.append(param.register_hook(hook))
        
        print(f'Z stage {self.stage_id} has {len(handles)} hooks', flush=True)
        
        return handles
    
    def uninstall_sync_hooks(self, handles):
        for handle in handles:
            handle.remove()

    def sync_grad(self):
        # ignore stage level allreduce
        if self.pctx.is_layer_allreduce:
            return

        nvtx.range_push('sync_grad' + self.nvtx_tag)
        
        grads = [p.grad for p in self.stage_module.parameters() if p.grad is not None]

        self._sync_grad_for_params_issue(grads)

        nvtx.range_pop()

    def wait_all(self):
        nvtx.range_push('sync_grad' + self.nvtx_tag)

        assert self.grad_sync_group is not None, 'grad_sync_group is not specified.'

        print(f'Z waiting for {len(self.handles)} handles', flush=True)

        for _ in range(len(self.handles)):
            self.handles.pop(0).wait()
            packed_tensor = self.packed_grads.pop(0) / self.grad_sync_group.size()
            vector_to_parameters(packed_tensor, self.grads.pop(0))

        nvtx.range_pop()

    def assert_intermediate_queues_are_empty(self):
        assert len(self.input_output_queue) == 0, f'input_output_queue of stage{self.stage_id} is not empty.'
        for name, queues in [('forward_send', self.forward_send_queues),
                             ('backward_recv', self.backward_recv_queues)]:
            for key, queue in queues.items():
                assert len(queue) == 0, f'{name}_queue for {key} of stage{self.stage_id} is not empty.'
    
    def _init_comm_queues(self):
        if not self.is_last_stage:
            for key in self.keys_for_next_stage:
                self.backward_recv_queues[key] = comm.Queue()
                self.forward_send_queues[key] = comm.Queue()
        if not self.is_first_stage:
            for key in self.keys_from_prev_stage:
                self.forward_recv_queues[key] = comm.Queue()
                self.backward_send_queues[key] = comm.Queue()

