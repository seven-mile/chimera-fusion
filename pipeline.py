import collections
from collections import deque
from typing import List, Tuple, Deque, OrderedDict, Iterator, Union, Dict
from contextlib import nullcontext

import torch
from torch import Tensor
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.distributed as dist
from torch.cuda import nvtx


import threading
from pipeline_context import PipelineContext
import threadsafe_queue
from chimera_pipeline_rank import ChimeraScheduleManager, CellType

PIPELINE_1F1B = '1f1b'
PIPELINE_GPIPE = 'gpipe'
PIPELINE_CHIMERA = 'chimera'
PIPELINE_INTER = 'interleave'


class StageModule(nn.Module):
    @property
    def keys_from_source(self) -> List[str]:
        raise NotImplementedError

    @property
    def sizes_from_prev_stage(self) -> Dict[str, Tuple]:
        raise NotImplementedError

    @property
    def sizes_for_next_stage(self) -> Dict[str, Tuple]:
        raise NotImplementedError


class GuardedThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        try:
            super().run()
        except Exception as e:
            print(f'Error in thread {self}: {e}', flush=True)


def start_comm_thread(func, kwargs):
    comm_thread = GuardedThread(target=func, kwargs=kwargs)
    comm_thread.daemon = True
    comm_thread.start()

class StageCommunicationManager:

    def __init__(self, device: torch.device):
        """
        Initialize the communication manager for the pipeline stage.

        Args:
            device (torch.device): Device to store tensors.
        """
        self.device = device

    @staticmethod
    def _recv_comm_thread(num_iterations, queue, src_rank, tag, tensor_shape, device):
        for _ in range(num_iterations):
            recv_tensor = torch.zeros(tensor_shape, requires_grad=True)
            if dist.get_backend() == dist.Backend.NCCL:
                recv_tensor = recv_tensor.to(device)
            dist.recv(tensor=recv_tensor, src=src_rank, tag=tag)
            queue.add(recv_tensor.to(device))

    @staticmethod
    def _send_comm_thread(num_iterations, queue, dst_rank, tag):
        for _ in range(num_iterations):
            send_tensor = queue.remove()
            if dist.get_backend() != dist.Backend.NCCL:
                send_tensor = send_tensor.cpu()
            
            dist.send(tensor=send_tensor, dst=dst_rank, tag=tag)

    def start_recv_threads(self, num_iterations, recv_queues, src_rank, tensor_shapes, tag):
        """
        Start threads for receiving tensors from the source rank.

        Args:
            num_iterations (int): Number of iterations to receive tensors.
            recv_queues (Dict[str, Queue]): Mapping the key of parameters to the queues to store their received tensors.
            src_rank (int): Source rank to receive tensors.
            tensor_shapes (Dict[str, Tuple[int]]): Shapes of tensors to receive, including batch size.
        """
        for key, queue in recv_queues.items():
            start_comm_thread(self._recv_comm_thread,
                                dict(num_iterations=num_iterations,
                                    queue=queue,
                                    src_rank=src_rank,
                                    tag=tag,
                                    tensor_shape=tensor_shapes[key],
                                    device=self.device))

    def start_send_threads(self, num_iterations, send_queues, dst_rank, tag):
        """
        Start threads for sending tensors to the destination rank.

        Args:
            num_iterations (int): Number of iterations to send tensors.
            send_queues (Dict[str, Queue]): Mapping the key of parameters to the queues to send their tensors.
            dst_rank (int): Destination rank to send tensors.
        """
        for queue in send_queues.values():
            start_comm_thread(self._send_comm_thread,
                                dict(num_iterations=num_iterations,
                                    queue=queue,
                                    dst_rank=dst_rank,
                                    tag=tag))


class Stage:

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
        if self.pctx.is_chimera:
            return self.prs_key + 1
        return 1

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

    def sync_grad(self):
        nvtx.range_push('sync_grad' + self.nvtx_tag)

        assert self.grad_sync_group is not None, 'grad_sync_group is not specified.'
        dist.barrier(group=self.grad_sync_group)
        grads = [p.grad for p in self.stage_module.parameters() if p.grad is not None]
        packed_tensor = parameters_to_vector(grads)
        dist.all_reduce(packed_tensor, group=self.grad_sync_group)
        packed_tensor /= self.grad_sync_group.size()
        vector_to_parameters(packed_tensor, grads)

        nvtx.range_pop()

    def wait_all(self):
        nvtx.range_push('wait_all' + self.nvtx_tag)

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
                self.backward_recv_queues[key] = threadsafe_queue.Queue()
                self.forward_send_queues[key] = threadsafe_queue.Queue()
        if not self.is_first_stage:
            for key in self.keys_from_prev_stage:
                self.forward_recv_queues[key] = threadsafe_queue.Queue()
                self.backward_send_queues[key] = threadsafe_queue.Queue()


class PipelineExecutor:

    def __init__(self,
                 pctx: PipelineContext,
                 stages: Dict[int, Stage],
                 data_iters: List[Iterator],
                 num_interations: int):
        """
        Initialize the pipeline executor.
        
        Args:
            pctx (PipelineContext): Pipeline context.
            stages (Dict[int, PipelineStage]): Dict of stage ids and their corresponding pipeline stages.
            data_iters (List[Iterator]): List of data iterators.
            num_interations (int): Number of iterations.
        """
        
        self.pctx = pctx
        self.stages = stages
        self.data_iters = data_iters
        self.num_interations = num_interations

        self.sched = self.pctx.sched_mgr.get_schedule(self.pctx.world_rank)

        self.comms = self._init_comm()

        for stage in self.stages.values():
            stage.stage_module.train()

    def _init_comm(self):
        # TODO: impl for interleaved
        if self.pctx.is_interleaved:
            raise NotImplementedError

        comms = {}
        for stage in self.stages.values():
            comm = StageCommunicationManager(self.pctx.device)

            def prepend_batch_sizes(shape_dict: Dict[str, Tuple]):
                return {key: self.pctx.batch_sizes + shape for key, shape in shape_dict.items()}

            comm.start_recv_threads(self.num_interations, stage.forward_recv_queues, stage.prev_rank, prepend_batch_sizes(stage.sizes_from_prev_stage), stage.p2p_tag)
            comm.start_send_threads(self.num_interations, stage.forward_send_queues, stage.next_rank, stage.p2p_tag)
            comm.start_recv_threads(self.num_interations, stage.backward_recv_queues, stage.next_rank, prepend_batch_sizes(stage.sizes_for_next_stage), stage.p2p_tag)
            comm.start_send_threads(self.num_interations, stage.backward_send_queues, stage.prev_rank, stage.p2p_tag)
            comms[stage.stage_id] = comm

        return comms
    
    def _assert_intermediate_queues_are_empty(self):
        """
        Assert that all intermediate queues are empty.
        """
        for stage in self.stages.values():
            stage.assert_intermediate_queues_are_empty()

    def run(self, iteration: int = None):
        """
        Run the pipeline for one iteration.

        Args:
            iteration (int): Iteration id.

        Returns:
            float: Total loss of the pipeline.
        """

        nvtx.range_push('call_pipeline')

        self._assert_intermediate_queues_are_empty()

        for cell in self.sched:
            if cell.is_idle():
                continue

            print(f'Z scheduling cell {cell}', flush=True)

            stage = self.stages[cell.stage_id]
            # print(f'Z communication prev_rank {stage.prev_rank} next_rank {stage.next_rank}', flush=True)
            if cell.type == CellType.FORWARD:
                stage.call_forward(next(self.data_iters[cell.pipeline_id]))
            elif cell.type == CellType.BACKWARD:
                stage.call_backward()
            elif cell.type == CellType.SYNC:
                stage.sync_grad()

        # TODO: no need for sync_grad
        for stage in self.stages.values():
            stage.wait_all()

        self._assert_intermediate_queues_are_empty()

        nvtx.range_pop()

        return stage.total_loss
