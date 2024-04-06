import torch
import torch.distributed as dist

import threading

class _GuardedThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        try:
            super().run()
        except Exception as e:
            print(f'Error in thread {self}: {e}', flush=True)


def _start_comm_thread(func, kwargs):
    comm_thread = _GuardedThread(target=func, kwargs=kwargs)
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
            _start_comm_thread(self._recv_comm_thread,
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
            _start_comm_thread(self._send_comm_thread,
                                dict(num_iterations=num_iterations,
                                    queue=queue,
                                    dst_rank=dst_rank,
                                    tag=tag))
