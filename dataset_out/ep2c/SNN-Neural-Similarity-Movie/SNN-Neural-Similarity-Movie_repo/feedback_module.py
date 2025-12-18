## feedback_module.py
import torch
import torch.nn as nn

class FeedbackModule(nn.Module):
    """
    Feedback module modeling corticocortical long-range feedback with recurrence and delay.
    Processes higher-region responses to generate feedback signals to lower layers.
    """

    def __init__(self, in_channels: int = 64,            # Feature dimension of higher region responses
                 feedback_strength: float = 1.0,        # Scalar to scale feedback signal
                 feedback_delay: int = 2,                 # Delay in time steps (corresponds to ms/frames)
                 feedback_connection_type: str = "recurrent"  # Feedback type; default 'recurrent'
                ):
        """
        Initialize FeedbackModule with configuration parameters.

        Args:
            in_channels (int): Dimension of higher-region response features.
            feedback_strength (float): Feedback influence scale.
            feedback_delay (int): Number of steps of delay in feedback connection.
            feedback_connection_type (str): Type of feedback ('recurrent' supported).
        """
        super().__init__()
        self.in_channels = in_channels
        self.feedback_strength = feedback_strength
        self.feedback_delay = feedback_delay
        self.feedback_connection_type = feedback_connection_type

        # Learnable linear projection for feedback
        self.feedback_proj = nn.Linear(in_channels, in_channels)
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.feedback_proj.weight)

        # Buffer to store delayed higher-region responses
        # Size: feedback_delay + 1 to include current response
        self.register_buffer("response_buffer", None)

    def reset(self, batch_size: int, device: torch.device):
        """
        Reset the feedback buffer at the start of a new stimulus/trial.
        Args:
            batch_size (int): Batch size for buffer initialization.
            device (torch.device): Device to place buffer tensors.
        """
        # Initialize buffer with zeros for delay
        self.response_buffer = [torch.zeros(batch_size, self.in_channels, device=device) for _ in range(self.feedback_delay + 1)]

    def forward(self, higher_response: torch.Tensor, current_step: int):
        """
        Generate feedback signal from higher-region response, accounting for delay.

        Args:
            higher_response (torch.Tensor): shape [batch_size, in_channels]
            current_step (int): current time step index in sequence

        Returns:
            feedback_signal (torch.Tensor): shape [batch_size, in_channels]
        """
        # Append current higher response to buffer
        if self.response_buffer is None:
            # On first call, initialize buffer
            batch_size = higher_response.shape[0]
            self.response_buffer = [torch.zeros(batch_size, self.in_channels, device=higher_response.device)
                                    for _ in range(self.feedback_delay + 1)]

        # Update buffer with current response
        self.response_buffer.append(higher_response.detach())

        # Handle delay: get response from 'feedback_delay' steps ago
        if current_step >= self.feedback_delay:
            delayed_response = self.response_buffer[-(self.feedback_delay + 1)]
        else:
            # Not enough history; use zeros
            delayed_response = self.response_buffer[0]

        # Remove oldest entry to keep buffer size consistent
        self.response_buffer.pop(0)

        # Project delayed response via learned weights
        feedback_signal = self.feedback_proj(delayed_response)
        # Scale feedback with strength parameter
        feedback_signal = self.feedback_strength * feedback_signal

        return feedback_signal
