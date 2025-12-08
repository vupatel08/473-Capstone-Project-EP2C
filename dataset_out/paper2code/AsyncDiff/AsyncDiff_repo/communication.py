## communication.py

import torch
import torch.distributed as dist
from typing import List

class Communication:
    """
    Handles inter-device communication primitives for AsyncDiff.
    Provides methods for broadcasting and gathering hidden states
    across multiple GPUs using NCCL backend.
    """

    def __init__(self):
        """
        Initialize the Communication class.
        Assumes torch.distributed has been initialized externally.
        """
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed is not initialized. "
                               "Please initialize torch.distributed before using Communication.")
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

    def broadcast_hidden_state(self, state: torch.Tensor, device_ids: List[int]) -> None:
        """
        Broadcasts a hidden state tensor from the device corresponding to '
        'device_ids[0]' (root) to all other devices in the process group.
        
        Args:
            state (torch.Tensor): The tensor to broadcast. Should be on the source device.
            device_ids (List[int]): List of device indices involved in the broadcast.
                                      The first device (device_ids[0]) is the root.
        """
        # Determine root device in device_ids
        root_rank = None
        # Find the index of the current rank in device_ids
        if self.rank in device_ids:
            root_rank = device_ids.index(self.rank)
        else:
            # Current process's device not in involved devices, raise error
            raise ValueError(f"Current process with rank {self.rank} not in device_ids {device_ids}.")

        # Ensure tensor is on the correct device for broadcasting
        # We assume state is already on the device corresponding to self.rank
        # Make sure the tensor is contiguous
        state = state.contiguous()

        # Broadcast tensor
        dist.broadcast(tensor=state, src=root_rank, group=None)
        # Note:
        # 'group=None' uses default process group
        # 'src' is index within device_ids, but 'dist.broadcast' uses rank within global process group
        # To handle multiple GPUs per process, ensure each process corresponds to one device
        # and that device_ids aligns with process ranks.

    def gather_hidden_states(self, states: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Gather hidden state tensors from all devices into a list.
        Each device supplies its local tensor; all tensors are collected.

        Args:
            states (List[torch.Tensor]): List of local tensors, length should be equal to world_size.

        Returns:
            List[torch.Tensor]: List of tensors from all devices ordered by device rank.
        """
        # Pre-allocate list for all gathered tensors
        gather_list = [torch.empty_like(states[0]) for _ in range(self.world_size)]
        # For consistent behavior, all processes share their local state
        # and gather into 'gather_list'
        dist.all_gather(gather_list, states[self.rank])
        return gather_list
