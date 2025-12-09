## trainer.py
import torch
import torch.nn.functional as F
import os
from typing import Dict, Any, Tuple, List
import numpy as np
from utils import (
    schedule_learning_rate,
    schedule_hyperparameters,
    compute_hessian_eigenvalue,
    estimate_hessian_eigenvalue,
    compute_zeroth_order_sharpness,
    compute_first_order_flatness,
    plot_loss_landscape,
    set_seed,
)

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 dataset_loader: 'DatasetLoader',
                 config: Dict[str, Any],
                 device: torch.device,
                 output_dir: str = './logs',
                 seed: int = 1993):
        """
        Initializes the Trainer with model, data, hyperparameters, and environment.

        Args:
            model (torch.nn.Module): The neural network model to train.
            dataset_loader (DatasetLoader): Loader for datasets and tasks.
            config (dict): Configuration dictionary with hyperparameters.
            device (torch.device): Computing device ('cuda' or 'cpu').
            output_dir (str): Directory for logs and checkpoints.
            seed (int): Random seed for reproducibility.
        """
        self.model = model.to(device)
        self.dataset_loader = dataset_loader
        self.device = device
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.seed = seed
        set_seed(self.seed)

        # Extract hyperparameters from config
        training_conf = config.get('training', {})
        model_conf = config.get('model', {})
        reg_conf = config.get('regularization', {})
        eval_conf = config.get('evaluation', {})
        self.lr = training_conf.get('learning_rate', 0.1)
        self.batch_size = training_conf.get('batch_size', 64)
        self.epochs = training_conf.get('epochs', 150)
        self.schedule_type = training_conf.get('schedule', {}).get('decay_type', 'exponential')
        self.decay_rate = training_conf.get('schedule', {}).get('decay_rate', 0.1)
        self.milestones = training_conf.get('schedule', {}).get('milestones', [])
        self.decay_factor = training_conf.get('schedule', {}).get('decay_factor', 0.1)

        # Regularization hyperparameters
        self.rho_init = reg_conf.get('rho', 0.2)
        self.lambda_init = reg_conf.get('lambda', 0.5)
        self.reg_eval_per_epoch = reg_conf.get('neighborhood_eval_per_epoch', 1)

        # Landscape visualization frequency
        self.landscape_eval_freq = eval_conf.get('landscape_visualization', False)
        self.metrics_on = eval_conf.get('metrics', True)

        # For hyperparameter scheduling
        self.current_rho = self.rho_init
        self.current_lambda = self.lambda_init

        # Initialize optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            milestones=classification= self.dataset_loader.get_task_dataloader.__func__.__annotations__.get('milestones', []),
                                                gamma=self.decay_factor)

        # Or setup custom scheduler
        if self.schedule_type == 'exponential':
            # Use step-based decay manually if desired
            pass

        # Statistics
        self.best_acc = 0.0
        self.training_history = []

    def train_phase(self, task_idx: int):
        """
        Train the model on the T-th task.

        Args:
            task_idx (int): index of current task.
        """
        dataloader = self.dataset_loader.get_task_dataloader(task_idx)
        total_step = len(dataloader) * self.epochs

        # Save initial model parameters for neighborhood calculations
        base_theta = [p.clone().detach() for p in self.model.parameters()]

        for epoch in range(1, self.epochs + 1):
            # Schedule learning rate
            lr = schedule_learning_rate(self.lr, epoch, {'decay_type': self.schedule_type,
                                                          'decay_rate': self.decay_rate,
                                                          'milestones': self.milestones})
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Schedule hyperparameters (\(\rho\), \(\lambda\))
            self.current_rho, self.current_lambda = schedule_hyperparameters(
                self.rho_init, self.lambda_init, epoch, self.epochs)

            # Optional: decay or update hyperparameters over epochs
            # e.g., exponential decay already handled in schedule_hyperparameters

            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Compute current gradient w.r.t. loss
                self.model.train()
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                task_loss = F.cross_entropy(outputs, labels)
                task_loss.backward(retain_graph=True)

                # Initialize regularization terms
                R0 = 0.0
                R1 = 0.0

                # Schedule regularizer evaluation: per epoch or batch
                eval_regularizer = False
                if self.reg_eval_per_epoch == 1:
                    # Evaluate once per epoch, so skip per batch
                    eval_regularizer = (batch_idx == 0)
                else:
                    # Per batch evaluation
                    eval_regularizer = True

                if eval_regularizer:
                    # Compute neighborhood regularizers

                    # 1. Zeroth-order sharpness regularization R0
                    # Get gradient
                    grads = [p.grad.clone() for p in self.model.parameters() if p.requires_grad]
                    grad_vec = []
                    for g in grads:
                        grad_vec.append(g.view(-1))
                    grad_concat = torch.cat(grad_vec)
                    grad_norm = torch.norm(grad_concat) + 1e-8
                    # Direction for perturbation
                    direction_params = []
                    for g in grads:
                        d = self.current_rho * g / (grad_norm)
                        direction_params.append(d)

                    # Save original params
                    orig_params = [p.clone() for p in self.model.parameters() if p.requires_grad]

                    # Perturb parameters for Rho^0
                    for p, d in zip(self.model.parameters(), direction_params):
                        if p.requires_grad:
                            p.data.add_(d)

                    # Compute loss at perturbed params
                    outputs_perturbed = self.model(inputs)
                    loss_perturbed = F.cross_entropy(outputs_perturbed, labels)

                    # Compute current unperturbed loss
                    self.model.zero_grad()
                    outputs = self.model(inputs)
                    loss_unperturbed = F.cross_entropy(outputs, labels)
                    R0 = (loss_perturbed - loss_unperturbed).item()

                    # Restore original parameters
                    for p, orig in zip(self.model.parameters(), orig_params):
                        p.data.copy_(orig.data)

                    # 2. First-order flatness regularization R1
                    # Approximate neighborhood gradient norm via Hessian-vector product
                    # Optionally, compute gradient norm at the perturbation point
                    # Gradient at current point
                    grads = [p.grad.clone() for p in self.model.parameters() if p.requires_grad]
                    grad_vec = []
                    for g in grads:
                        grad_vec.append(g.view(-1))
                    grad_concat = torch.cat(grad_vec)
                    grad_norm = torch.norm(grad_concat) + 1e-8

                    # Perturb along gradient
                    direction_params = []
                    for g in grads:
                        d = self.current_rho * g / (grad_norm)
                        direction_params.append(d)

                    # Save original params
                    orig_params = [p.clone() for p in self.model.parameters() if p.requires_grad]
                    # Perturb
                    for p, d in zip(self.model.parameters(), direction_params):
                        if p.requires_grad:
                            p.data.add_(d)

                    # Compute neighborhood gradient norm (second evaluation)
                    outputs_perturbed = self.model(inputs)
                    loss_perturbed = F.cross_entropy(outputs_perturbed, labels)

                    # Compute gradient again at perturbed point
                    self.model.zero_grad()
                    outputs = self.model(inputs)
                    loss_at_perturbed = F.cross_entropy(outputs, labels)
                    # Compute gradient w.r.t. parameters
                    grads_perturbed = torch.autograd.grad(loss_at_perturbed, self.model.parameters(), create_graph=True)
                    g_p = []
                    for g in grads_perturbed:
                        g_p.append(g.view(-1))
                    g_p_concat = torch.cat(g_p)
                    R1 = self.current_rho * torch.max(torch.norm(g_p_concat))
                    
                    # For simplicity, approximate R1 as max gradient norm (could be refined)

                    # Restore params
                    for p, orig in zip(self.model.parameters(), orig_params):
                        p.data.copy_(orig.data)

                # Now, compute total loss with regularizers
                # Recompute task loss for backprop
                self.model.zero_grad()
                outputs = self.model(inputs)
                task_loss = F.cross_entropy(outputs, labels)

                # Convert R0 and R1 to tensors
                R0_tensor = torch.tensor(R0, device=self.device)
                R1_tensor = torch.tensor(R1, device=self.device)
                total_loss = task_loss + R0_tensor + self.current_lambda * R1_tensor

                # Backpropagate total loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            # Scheduler step
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                self.scheduler.step()

            # Post-epoch: optional landscape evaluation
            if self.landscape_eval_freq > 0 and epoch % self.landscape_eval_freq == 0:
                self._evaluate_landscape(task_idx, epoch)

        # After all epochs in a phase, save model checkpoint
        checkpoint_path = os.path.join(self.output_dir, f'model_task_{task_idx}.pt')
        torch.save(self.model.state_dict(), checkpoint_path)

    def evaluate(self, task_idx: int) -> Dict[str, float]:
        """
        Evaluate model on all seen classes (or test set).

        Args:
            task_idx (int): current task index.
        Returns:
            dict: evaluation metrics such as accuracy
        """
        # Prepare test loader
        test_loader = self.dataset_loader.get_full_test_dataset()
        test_loader = self.dataset_loader.get_test_dataloader(batch_size= self.batch_size, shuffle=False)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total * 100
        return {'accuracy': acc}

    def _evaluate_landscape(self, task_idx: int, epoch: int):
        """
        Compute Hessian eigenvalues and visualize loss landscape periodically.
        """
        if not hasattr(self, 'model') or not hasattr(self, 'dataset_loader'):
            return  # Safety

        # Get current model parameters
        model_params = [p.clone().detach() for p in self.model.parameters()]
        # Calculate Hessian eigenvalue
        # Use a small batch from current data or validation data
        # For simplicity, re-use last batch in last epoch, or create a small batch
        # Here, placeholder: skip actual inputs
        # User can implement with validation data
        # Example: skip if inputs not available
        try:
            # Fake data: should be replaced with actual validation data
            inputs = torch.randn(16, 3, 32, 32).to(self.device)
            labels = torch.randint(0, 100, (16,)).to(self.device)
            eig_val = compute_hessian_eigenvalue(self.model, F.cross_entropy, inputs, labels, num_iterations=20)
        except Exception:
            eig_val = None

        # Save or print eigenvalue
        print(f"Epoch {epoch} Task {task_idx}: Hessian max eigenvalue: {eig_val}")

        # Visualize the loss landscape along two random directions
        # Generate random directions
        dir1 = [torch.randn_like(p) for p in self.model.parameters()]
        dir2 = [torch.randn_like(p) for p in self.model.parameters()]

        # Normalize directions
        def flatten_params(p_list):
            return torch.cat([p.view(-1) for p in p_list])
        v1 = flatten_params(dir1)
        v2 = flatten_params(dir2)
        v1 /= torch.norm(v1) + 1e-8
        v2 /= torch.norm(v2) + 1e-8

        # Plot landscape
        # Note: the method expects list of tensors for directions, adapt visualization accordingly
        plot_loss_landscape(self.model,
                            F.cross_entropy,
                            directions=(v1, v2),
                            center_params=[p.clone().detach() for p in self.model.parameters()],
                            grid_size=50,
                            depth=1.0)  # or adaptive depth

    def run(self):
        """
        Run the full training over all tasks.
        """
        total_tasks = len(self.dataset_loader.get_class_sets())
        for t in range(total_tasks):
            print(f"Starting training for task {t+1}/{total_tasks}")
            self.train_phase(t)
            print(f"Completed training for task {t+1}/{total_tasks}")
            # Evaluate on all seen tasks so far
            metrics = self.evaluate(t)
            print(f"Evaluation metrics after task {t+1}: {metrics}")
            # Save metrics for later analysis if needed
