## train.py
import torch
import time
from utils import set_seeds, save_checkpoint, plot_training_curve, normalize_input
from dataset import DatasetLoader
from model import Model
from optimizer import ScheduleFreeOptimizer

class TrainLoop:
    def __init__(self, config: dict):
        # Set random seed for reproducibility
        seed = config.get('training', {}).get('seed', 42)
        set_seeds(seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load dataset
        dataset_cfg = config.get('dataset', {})
        dataset_name = dataset_cfg.get('name', 'CIFAR10')
        data_path = dataset_cfg.get('data_path', './data')
        batch_size = dataset_cfg.get('batch_size', 32)
        self.train_loader = DatasetLoader(
            dataset_name=dataset_name,
            data_path=data_path,
            batch_size=batch_size,
            train=True
        ).load_data()
        self.val_loader = DatasetLoader(
            dataset_name=dataset_name,
            data_path=data_path,
            batch_size=batch_size,
            train=False
        ).load_data()

        # Initialize model
        model_cfg = config.get('model', {})
        architecture = model_cfg.get('architecture', 'ResNet50')
        hyperparams = {k: v for k, v in model_cfg.items() if k != 'architecture'}
        self.model = Model(architecture, hyperparams).get_model().to(self.device)

        # Hyperparameters
        training_cfg = config.get('training', {})
        self.num_epochs = training_cfg.get('epochs', 100)
        self.batch_size = training_cfg.get('batch_size', 32)
        warmup_steps = training_cfg.get('warmup_steps', 4000)
        large_lr_flag = training_cfg.get('large_learning_rate', True)
        initial_lr = training_cfg.get('learning_rate', 0.0025)  # default fixed large lr

        # Estimate bounds D and G
        # Here, we set D and G as per prior knowledge or estimation.
        D = training_cfg.get('initial_D', 1.0)  # Placeholder, can estimate
        G = training_cfg.get('G_estimate', 1.0)  # Placeholder, can estimate from data

        # Compute fixed large learning rate based on D/G ratios
        if large_lr_flag:
            self.gamma = D / G  # constr. from theory D/G; fallback to manual if needed
        else:
            self.gamma = initial_lr

        # Hyperparameters for optimizer
        beta = training_cfg.get('beta', 0.9)
        weight_decay = training_cfg.get('weight_decay', 1e-4)
        # Initialize optimizer
        self.optimizer = ScheduleFreeOptimizer(
            model_params=list(self.model.parameters()),
            optimizer_type=training_cfg.get('optimizer', 'AdamW'),
            lr_scale=1.0,  # scale will be set as self.gamma
            beta=beta,
            D=D,
            G=G,
            eta=self.gamma,
            weight_decay=weight_decay
        )

        # Training tracking
        self.global_step = 0
        self.log_interval = config.get('logging', {}).get('log_interval', 50)  # steps
        self.checkpoint_dir = config.get('logging', {}).get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.metrics_history = {'loss': [], 'accuracy': [], 'lr': []}
        self.loss_fn = torch.nn.CrossEntropyLoss()  # default loss, modify if needed

    def train(self):
        total_steps = 0
        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            epoch_loss = 0.0
            epoch_correct = 0
            total_samples = 0
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs = normalize_input(inputs).to(self.device)
                labels = labels.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()

                # Step the Schedule-Free optimizer
                self.optimizer.step(gradient_eval_fn=self._compute_gradients_at_y)

                # Metrics
                batch_loss = loss.item()
                epoch_loss += batch_loss * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct = (preds == labels).sum().item()
                epoch_correct += correct
                total_samples += inputs.size(0)

                self.global_step += 1
                total_steps += 1

                # Logging
                if self.global_step % self.log_interval == 0:
                    avg_loss = epoch_loss / total_samples
                    accuracy = epoch_correct / total_samples
                    print(f"Epoch {epoch} Step {self.global_step}: Loss={avg_loss:.4f} "
                          f"Accuracy={accuracy:.4f} LR={self.optimizer.eta:.6f}")
                    self.metrics_history['loss'].append(avg_loss)
                    self.metrics_history['accuracy'].append(accuracy)
                    self.metrics_history['lr'].append(self.optimizer.eta)
                    # Save checkpoint
                    checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_step_{self.global_step}.pt')
                    save_checkpoint(self.model, self.optimizer.optimizer, checkpoint_path)

            # Epoch end metrics
            epoch_loss_avg = epoch_loss / total_samples
            epoch_accuracy = epoch_correct / total_samples
            print(f"Epoch {epoch} completed: Loss={epoch_loss_avg:.4f} "
                  f"Accuracy={epoch_accuracy:.4f}")

        total_time = time.time() - start_time
        print(f"Training completed in {total_time/60:.2f} minutes.")

        # Final evaluation on validation set using the last interpolation x_T
        final_metrics = self._evaluate(self.val_loader)
        print(f"Validation results: {final_metrics}")

        # Save final model parameters (x_T)
        # Retrieve x_T parameters
        x_T_params = self.optimizer.get_current_x_params()
        self._load_params_into_model(self.model, x_T_params)
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, 'final_x_T.pt'))

        return self.metrics_history, final_metrics

    def _compute_gradients_at_y(self, y_params, inputs=None, labels=None):
        """
        Compute gradients at evaluation point y_t.
        y_params: list of tensors representing the evaluation point.
        """
        # Assign y_t to model parameters
        for p, y_p in zip(self.model.parameters(), y_params):
            p.data.copy_(y_p)
        # Zero gradients
        self.optimizer.optimizer.zero_grad()
        # Forward pass
        outputs = self.model(inputs) if inputs is not None else None
        # If no inputs/labels provided, need to pass data outside
        # but for simplicity, assume inputs/labels provided
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        # Collect gradients
        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                grads.append(p.grad.clone())
            else:
                grads.append(torch.zeros_like(p))
        return grads

    def _load_params_into_model(self, model, params_list):
        """
        Assign parameters in params_list into model in-place.
        """
        for p, new_p in zip(model.parameters(), params_list):
            p.data.copy_(new_p)

# Example of instantiating and running training
if __name__ == '__main__':
    import yaml
    import os
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    trainer = TrainLoop(config)
    metrics_history, final_metrics = trainer.train()
    # Optionally plot training curves
    plot_training_curve(metrics_history)
