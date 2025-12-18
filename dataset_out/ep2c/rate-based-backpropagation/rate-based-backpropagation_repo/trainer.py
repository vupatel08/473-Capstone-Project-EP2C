# trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import logging
import time
from utils import load_config, set_random_seed, save_config, initialize_logger
from dataset_loader import DatasetLoader
from model import ResNet, VGG, SEW_ResNet
from surrogate_gradients import sigmoid
from torch.utils.data import DataLoader

class RateBasedTrainer:
    """
    Implements the core training loop for deep SNNs with rate-based backpropagation,
    following the methodology described in the paper, including eligibility traces,
    surrogate gradients, and simplified gradient computation.
    """
    def __init__(self, config, device):
        """
        Initialize trainer with configuration and device.
        Args:
            config (dict): configuration parameters from YAML
            device (torch.device): CPU or CUDA
        """
        self.config = config
        self.device = device
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.criterion = nn.CrossEntropyLoss()
        self.T = self.config['training']['T']
        self.training_mode = self.config['training_mode']['mode']
        self.sequence_length = self.T
        self.num_layers = self._count_layers()
        self._init_training_state()
        self.logger = None  # will be initialized outside

    def _build_model(self):
        """
        Instantiate the neural network based on specified architecture and config.
        """
        arch = self.config['model']['architecture']
        if arch.lower() == 'resnet18':
            model = ResNet(architecture='resnet18', config=self.config, training_mode=self.training_mode)
        elif arch.lower() == 'vgg11':
            model = VGG('VGG11', num_classes=10)  # adapt size for dataset if needed
        elif arch.lower() == 'sew-resnet34':
            model = SEW_ResNet('sew-resnet34', config=self.config, training_mode=self.training_mode)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        model = model.to(self.device)
        return model

    def _build_optimizer(self):
        """
        Build optimizer with parameters from config.
        """
        optim_type = self.config['training'].get('optimizer', 'Adam')
        lr = self.config['training'].get('learning_rate', 0.1)
        wd = self.config['training'].get('weight_decay', 5e-4)
        if optim_type == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif optim_type == 'SGD':
            momentum = self.config['training'].get('momentum', 0.9)
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
        else:
            raise ValueError(f"Unsupported optimizer: {optim_type}")
        return optimizer

    def _count_layers(self):
        """
        Count trainable layers, mainly the number of weight layers for gradient accumulation.
        """
        count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                count += 1
        return count

    def _init_training_state(self):
        """
        Initialize eligibility traces and auxiliary variables for each layer.
        """
        self.e_trace = {}  # e_t^l: eligibility trace
        self.g_trace = {}  # g_t^l: gradient estimators
        self.rho = {}      # rho_t^l: neuron dynamics influence
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                size = param.shape
                self.e_trace[name] = torch.zeros(size, device=self.device)
                self.g_trace[name] = torch.zeros(size, device=self.device)
                # For rho, shape matches the neuron layer output
                self.rho[name] = 0.0  # scalar or tensor as appropriate
        # Might initialize more if needed for batch norm statistics, etc.

    def set_logger(self, log_path):
        """
        Initialize logger for logging training metrics.
        """
        self.logger = initialize_logger(log_path)

    def train(self, train_loader, val_loader, num_epochs, save_dir):
        """
        Main training loop over epochs and data.
        Args:
            train_loader (DataLoader): training data loader
            val_loader (DataLoader): validation data loader
            num_epochs (int): total epochs
            save_dir (str): directory for saving models and logs
        """
        os.makedirs(save_dir, exist_ok=True)
        start_time = time.time()
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            self.model.train()
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                batch_time_start = time.time()
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                # Reset eligibility traces and neuron states
                self._reset_traces()
                # Forward pass
                output, rate_activations, neuron_states = self._forward_pass(inputs, batch_size=inputs.size(0))
                # Compute loss
                loss = self.criterion(output, targets)
                total_loss += loss.item() * inputs.size(0)
                # Backward for rate-based gradient
                self.optimizer.zero_grad()
                # Compute rate gc
                d_L_d_rate_prop = self._compute_output_gradient(rate_activations, targets)
                # Backpropagation through the network
                self._backward(rate_activations, neuron_states, d_L_d_rate_prop)
                # Optimizer step
                self.optimizer.step()
                # Compute accuracy
                preds = output.argmax(dim=1)
                correct = (preds == targets).sum().item()
                total_correct += correct
                total_samples += inputs.size(0)
                batch_time_end = time.time()

                if self.logger and batch_idx % 50 == 0:
                    self.logger.info(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                                     f"Loss: {loss.item():.4f} Acc: {correct/inputs.size(0):.4f} "
                                     f"Time: {batch_time_end - batch_time_start:.2f}s")
            epoch_time = time.time() - epoch_start
            train_loss_avg = total_loss / total_samples
            train_acc = total_correct / total_samples
            self._validate(val_loader)
            self._save_checkpoint(os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt"))
            if self.logger:
                self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s, "
                                 f"Avg Loss: {train_loss_avg:.4f}, Accuracy: {train_acc:.4f}")
            # decay schedule if any
            self._lr_schedule(epoch)
        total_time = time.time() - start_time
        if self.logger:
            self.logger.info(f"Training completed in {total_time/60:.2f} minutes.")

    def _reset_traces(self):
        """
        Zero out eligibility and auxiliary traces for the new batch.
        """
        for key in self.e_trace:
            self.e_trace[key].zero_()
            self.g_trace[key].zero_()
            if isinstance(self.rho[key], torch.Tensor):
                self.rho[key].fill_(0.0)
            else:
                self.rho[key] = 0.0

    def _forward_pass(self, inputs, batch_size):
        """
        Implement the forward pass with rate-based approximation, eligibility trace updates.
        Args:
            inputs (Tensor): input data tensor with shape [B, T, C, H, W] or [B, C, H, W] for rate_S
            batch_size (int): batch size
        Returns:
            output (Tensor): class logits
            rate_activations (dict): per-layer rate estimates
            neuron_states (dict): stored neuron membrane potentials and spikes for backward
        """
        # Initialize the rate dictionaries
        rate_activations = {}
        neuron_states = {}  # to store u, s, e_t, g_t, rho during the sequence
        # Prepare initial inputs for the network
        if self.training_mode == 'rate_M':
            # inputs shape: [B, T, C, H, W]
            # Initialize containers
            batch = inputs.shape[0]
            # initialize memory states
            # For each layer, we keep u, s states, eligibility traces etc.
            # For simplicity, here we implement a minimal version. More detailed per-layer storage is recommended.
            # --- Start forward pass for T steps ---
            # For illustration, assume inputs are [B, T, C, H, W], sequence dimension T
            outputs_list = []
            # Initialize neuron states and eligibility traces for all layers
            neuron_state_buffers = {}  # e.g., {'layer1': {'u': ..., 's': ...}}
            for name, layer in self.model.named_modules():
                if hasattr(layer, 'neuron1'):
                    neuron_state_buffers[name] = {
                        'u': torch.zeros(batch_size, layer.neuron1.V_th.shape[1], device=self.device),
                        's': torch.zeros(batch_size, layer.neuron1.V_th.shape[1], device=self.device)
                    }
            # Loop over T to process each timestep
            for t in range(self.T):
                # Get input for time t
                x_t = inputs[:, t, ...]
                # Forward through network
                out, layer_states = self._forward_single_step(x_t, neuron_state_buffers)
                # Save/accumulate outputs for loss calculation
                outputs_list.append(out)
                # Update neuron states for next timestep
                # Store states internally, or in buffer
            # Compute average output
            output_logits = torch.stack(outputs_list, dim=1).mean(dim=1)
            # Prepare the rate activations dictionary (if needed for loss)
            rate_activations['layer_outputs'] = output_logits
            neuron_states['layer_neurons'] = layer_states
        else:
            # Rate_S: single step, process one timestep
            x_t = inputs  # shape: [B, C, H, W]
            out, layer_states = self._forward_single_step(x_t, None)
            output_logits = out
            rate_activations['layer_outputs'] = output_logits
            neuron_states['layer_neurons'] = layer_states

        return output_logits, rate_activations, neuron_states

    def _forward_single_step(self, x_t, neuron_state_buffers):
        """
        Forward function for a single timestep.
        Args:
            x_t (Tensor): current input
            neuron_state_buffers (dict): neuron states for each layer, if any
        Returns:
            out: network output at this timestep
            layer_states: store neuron states for backprop
        """
        # Forward through initial conv + BN + neuron
        # Placeholder: use model defined in 'model.py'
        # For illustration:
        # Replace with actual calls to model modules and neuron update steps
        out = x_t
        layer_states = {}
        # Example: process through model layers, updating neuron states and eligibility traces
        # For each layer, implement update of u, s
        # one must integrate surrogate gradients, eligibility trace update, etc.
        # Since this code is highly schematic, assume the model handles internal states
        # In actual implementation, call model forward with appropriate mode
        # For now, return the raw input as output (to be replaced with real model forward)
        return out, layer_states

    def _compute_output_gradient(self, rate_outputs, targets):
        """
        Compute the gradient of the loss w.r.t. the rate outputs (e.g., via loss derivative)
        Placeholder: this depends on actual loss implementation.
        Args:
            rate_outputs (dict): network output estimates
            targets (Tensor): true labels
        Returns:
            d_L_d_rate (Tensor): derivative of loss with respect to rate estimates
        """
        # For simplicity, assuming loss is cross-entropy, so:
        # Compute softmax and derivative
        logits = rate_outputs['layer_outputs']
        probs = nn.functional.softmax(logits, dim=1)
        grad = probs
        grad.scatter_(1, targets.unsqueeze(1), 0)
        grad = -grad / logits.shape[0]  # normalized derivative
        return grad

    def _backward(self, rate_activations, neuron_states, d_L_d_rate):
        """
        Perform backward pass for rate-based gradients using eligibility traces and surrogate derivatives.
        Args:
            rate_activations (dict): stored rate estimates
            neuron_states (dict): stored neuron mem potentials and spikes
            d_L_d_rate (Tensor): gradient of loss w.r.t. rate output
        """
        # Implement a simplified backward as per rate-based derivation in paper
        # E.g., compute delta terms (δ) for each layer
        # Update gradients of W^l as: ΔW^l = δ^l * (r^{l-1})^T
        # For each layer, compute δ and gradients
        # Here, we demonstrate a minimal example;
        # the actual implementation should match the gradient derivations and variable storage.
        for name, param in self.model.named_parameters():
            # Skip bias or non-weight parameters
            if 'weight' not in name:
                continue
            # For illustration, generate dummy delta
            delta = torch.ones_like(param.data)  # Placeholder
            # Compute gradient as outer product with previous layer rate
            prev_rate = rate_activations.get('layer_outputs', None)
            if prev_rate is None:
                continue
            grad_W = torch.mm(delta.view(-1,1), prev_rate.view(1,-1))
            param.grad = grad_W

        # After computing gradients, optimizer steps will update weights

    def _save_checkpoint(self, filepath):
        """
        Save model parameters, optimizer state, and current trace state.
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def _load_checkpoint(self, filepath):
        """
        Load previous checkpoint.
        """
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def _lr_schedule(self, epoch):
        """
        Implement decay schedule, e.g., exponential per epoch.
        """
        base_lr = self.config['training'].get('learning_rate', 0.1)
        decay_rate = self.config['training'].get('decay_rate', 0.95)
        new_lr = base_lr * (decay_rate ** (epoch - 1))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def _validate(self, val_loader):
        """
        Run validation on the validation set.
        """
        total_correct = 0
        total_samples = 0
        self.model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output = self._forward_inference(inputs)
                preds = output.argmax(dim=1)
                total_correct += (preds == targets).sum().item()
                total_samples += inputs.size(0)
        accuracy = total_correct / total_samples
        if self.logger:
            self.logger.info(f"Validation accuracy: {accuracy:.4f}")

    def _forward_inference(self, inputs):
        """
        Forward pass during inference with the standard mode.
        """
        # Simplified: just run the model's direct inference
        return self.model(inputs, mode='rate_S', T=1)

# Usage example outside this class:
# if __name__ == '__main__':
#     config = load_config('config.yaml')
#     set_random_seed(config['training'].get('seed', 42))
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     trainer = RateBasedTrainer(config, device)
#     train_loader, val_loader = DatasetLoader(...).load_data()
#     trainer.set_logger(os.path.join(config['logging']['log_dir'], 'train_log.txt'))
#     trainer.train(train_loader, val_loader, num_epochs=config['training']['epochs'], save_dir='./checkpoints')
