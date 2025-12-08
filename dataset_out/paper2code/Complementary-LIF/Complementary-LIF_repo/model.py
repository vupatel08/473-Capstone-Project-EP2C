## model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuron import CLIFunction

class CLIFLayer(nn.Module):
    """
    A wrapper for a single CLIF neuron layer (for convolutional or linear layers).
    Maintains internal states u (membrane potential) and m (complementary potential).
    """
    def __init__(self, size, V_th=1.0, tau=1.5):
        """
        Args:
            size (tuple): shape of the input (batch_size, channels, H, W) for conv, or (batch_size, features) for fc
            V_th (float): threshold voltage
            tau (float): membrane time constant
        """
        super().__init__()
        self.V_th = V_th
        self.tau = tau
        self.shape = size
        # Initialize states; these should be reset externally before each sequence
        self.register_buffer('u', torch.zeros(size))
        self.register_buffer('m', torch.zeros(size))
    
    def reset_state(self, batch_size):
        """
        Reset states for a new sequence/batch.
        """
        self.u = torch.zeros(self.shape, device=self.u.device)
        self.m = torch.zeros(self.shape, device=self.m.device)

    def forward(self, input_current):
        """
        Compute spike output s over a single timestep given current input.
        Args:
            input_current (Tensor): shape matching internal states
        Returns:
            s (Tensor): binary spike tensor
        """
        # Call the autograd function
        s = CLIFunction.apply(self.u, self.m, input_current, self.V_th, self.tau)
        # After forward, update internal state variables for next timestep
        # u and m are updated within CLIFunction; here, we assign for next iteration
        # (Assuming external code manages state updates, or we do here)
        # For batch processing, assign the updated states
        # Extract the last computed u and m from function context if needed
        # For simplicity, we assign directly: (this works if further managed externally)
        # Here, for a simple approach, we cache current states for next call; 
        # in training loop, user should call reset_state() and handle states
        return s

    def update_states(self, u_new, m_new):
        """
        Manually update states after computation for external control.
        """
        self.u = u_new
        self.m = m_new


class BasicConvBlock(nn.Module):
    """
    Basic convolutional block with conv + BatchNorm + CLIF neuron activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, V_th=1.0, tau=1.5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.V_th = V_th
        self.tau = tau
        # Placeholders for CLIF neuron; will be instantiated in create_layer
        self.neuron_layer = None

    def create_layer(self, size):
        self.neuron_layer = CLIFLayer(size, V_th=self.V_th, tau=self.tau)

    def reset_state(self, batch_size):
        if self.neuron_layer:
            self.neuron_layer.reset_state(batch_size)

    def forward(self, x):
        """
        Forward pass with CLIF activation.
        Args:
            x: input feature map (batch, channels, H, W)
        Returns:
            spike output (batch, channels, H, W)
        """
        out = self.conv(x)
        out = self.bn(out)
        # Initialize neuron states if not already
        if self.neuron_layer is None:
            self.create_layer(out.shape)
        if not hasattr(self.neuron_layer, 'u'):
            self.neuron_layer.reset_state(out.shape)
        # Use CLIF neuron
        s = self.neuron_layer(out)
        # Update neuron states after this timestep
        # It is the user's responsibility to call update_states() outside after each timestep
        return s

class ResidualBlock(nn.Module):
    """
    Basic residual block with two conv layers and CLIF activation, residual connection.
    """
    def __init__(self, in_channels, out_channels, stride=1, V_th=1.0, tau=1.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.V_th = V_th
        self.tau = tau

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # CLIF activations for each convolutional layer
        self.activation1 = None
        self.activation2 = None
        self.create_activation_layers()

    def create_activation_layers(self):
        # Create activation layers after convs
        sample_size = (1, self.out_channels, 32, 32)  # Example, will be reset during forward
        self.activation1 = CLIFLayer(sample_size, V_th=self.V_th, tau=self.tau)
        self.activation2 = CLIFLayer(sample_size, V_th=self.V_th, tau=self.tau)

    def reset_state(self, batch_size):
        # Reset internal states for activation layers
        if self.activation1:
            self.activation1.reset_state(batch_size)
        if self.activation2:
            self.activation2.reset_state(batch_size)

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        if self.activation1 is None:
            self.create_activation_layers()
        if not hasattr(self.activation1, 'u'):
            self.activation1.reset_state(out.shape)
        s1 = self.activation1(out)

        out2 = self.conv2(s1)
        out2 = self.bn2(out2)
        if self.activation2 is None:
            self.create_activation_layers()
        if not hasattr(self.activation2, 'u'):
            self.activation2.reset_state(out2.shape)
        s2 = self.activation2(out2)

        out = s2 + residual
        return out

class SpikingResNet(nn.Module):
    """
    ResNet-18-like architecture with CLIF neurons.
    """
    def __init__(self, num_classes=10, V_th=1.0, tau=1.5, T=6):
        super().__init__()
        self.V_th = V_th
        self.tau = tau
        self.T = T  # total timesteps
        # First conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Residual layers
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        # Final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
        # Initialize states
        self._initialize_states()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, V_th=self.V_th, tau=self.tau))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, V_th=self.V_th, tau=self.tau))
        return nn.Sequential(*layers)

    def _initialize_states(self):
        """
        Prepare for state management during sequence processing.
        """
        # For each residual block, reset states
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                block.reset_state(batch_size=1)  # reset with batch size 1; real batch sizes managed in training loop

    def reset_state(self, batch_size):
        """
        Reset states of all layers before processing a new sequence.
        """
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                block.reset_state(batch_size)
        # Also reset first conv if needed (not necessary)
        # No specific state variables for initial conv

    def forward(self, x):
        """
        Process input sequence over T timesteps.
        Args:
            x: shape [batch_size, T, C, H, W]
        Returns:
            logits (tensor): class logits at the end of sequence
        """
        batch_size, T, C, H, W = x.shape
        device = x.device
        # Reset states at start
        self.reset_state(batch_size)
        # Loop over timesteps
        for t in range(T):
            x_t = x[:, t]  # shape [batch, C, H, W]
            # First conv + bn
            out = self.conv1(x_t)
            out = self.bn1(out)
            # Activate through CLIF layer of first residual block
            layer1_block = self.layer1[0]
            if not hasattr(layer1_block.activation1, 'u'):
                layer1_block.activation1.reset_state(batch_size)
            s1 = layer1_block.activation1(out)
            # Forward through residual blocks
            s2 = s1
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for block in layer:
                    s2 = block(s2)  # each block manages its own states
        # After last time step, global average pooling
        out_feat = self.avgpool(s2)
        out_feat = out_feat.view(batch_size, -1)
        logits = self.fc(out_feat)
        return logits
