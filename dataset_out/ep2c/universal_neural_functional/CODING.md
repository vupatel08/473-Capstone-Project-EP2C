# Coding Phase

This document contains the code generation artifacts for each file in the implementation.

## basis_generator.py

```python
## basis_generator.py
import itertools
from typing import List, Tuple, Dict, Callable, Any
import string
import functools
import jax
import jax.numpy as jnp

# Import data structures
from architecture_spec import ArchitectureSpec, LayerSpec, TensorSpec, PermutationGroup

# Character labels for subsets
CHARACTERS = list(string.ascii_lowercase)

class ArrayOperation:
    """
    Represents an array operation (e.g., sum over certain indices)
    that can be applied to a tensor to produce a basis function.
    """

    def __init__(self, apply_fn: Callable[[jnp.ndarray], jnp.ndarray]):
        self.apply_fn = apply_fn

    def __call__(self, tensor: jnp.ndarray) -> jnp.ndarray:
        return self.apply_fn(tensor)


class BasisGenerator:
    """
    Implements Algorithm 1 for constructing a basis of equivariant linear maps
    between tensor spaces, based on their index partitions and symmetry constraints.
    """

    def __init__(self, architecture_spec: ArchitectureSpec):
        """
        Initializes the BasisGenerator with architecture specifications,
        extracting tensor shapes and symmetry information.
        """
        self.arch_spec = architecture_spec
        # Precompute a list of all layer tensor specs
        self.layer_tensor_specs: Dict[str, List[TensorSpec]] = {}
        for layer in self.arch_spec.layers:
            tensor_specs = []
            # Assuming `get_tensor_spec()` provides shape and symmetry info
            tensor_specs.extend(self.arch_spec.get_tensor_spec(layer.name))
            self.layer_tensor_specs[layer.name] = tensor_specs

    def generate_basis(self, tensor_spec: TensorSpec) -> List[ArrayOperation]:
        """
        Generates a list of basis array operations for a single tensor,
        according to the symmetry group acting on the tensor's indices.
        """
        shape = tensor_spec.shape  # e.g., (n_output, n_input, k)
        # Label axes with their symmetry info
        axis_labels = []
        # Determine active symmetry axes (permutation symmetry)
        # and their sizes, for each dimension
        # For simplicity, assume symmetry info matches shape length
        for i, size in enumerate(shape):
            axis_labels.append((i, size))

        # Determine which axes are permutable based on symmetry_type
        permutable_axes = []
        for i, (ax_idx, size) in enumerate(axis_labels):
            if tensor_spec.symmetry_type == 'permutation':
                # We assume all axes are permutation symmetric unless specified otherwise
                permutable_axes.append(ax_idx)
            else:
                # Non-permutable axes
                pass

        # Generate all valid partitions of the index set corresponding to axes
        all_partitions = list(self._enumerate_partitions(permutable_axes))

        basis_list: List[ArrayOperation] = []

        for partition in all_partitions:
            # Assign characters to subsets in the partition
            char_mapping = self._assign_characters_to_partition(partition)

            # Define the array operation E_P for this partition
            def make_operation(p=partition, c_mapping=char_mapping):
                def operation(tensor: jnp.ndarray) -> jnp.ndarray:
                    # tensor shape: shape
                    # Generate all indices for tensor
                    # For large tensors, avoid explicit looping
                    # instead, create masks based on label matching
                    index_grids = jnp.indices(shape)  # shape: (dim, shape...)
                    # index_grids: shape (D, shape), with D=rank
                    # For each axis, get indices
                    axis_indices = index_grids  # shape: (D, shape...)

                    # Map indices to labels
                    label_mask = None
                    for axis, label_char in c_mapping.items():
                        # For the axis, create a boolean mask where indices match label
                        match_mask = (axis_indices[axis] == label_char)
                        # Combine masks across axes
                        label_mask = match_mask if label_mask is None else label_mask & match_mask

                    # For each output index, sum over input indices where label masks match
                    # Here, consider the shape of output tensor matches 'shape'
                    # Create a mask for summation
                    # For simplicity, perform einsum over all axes
                    # Each basis function is a sum over grouped indices
                    # To vectorize, we can construct a weighting tensor
                    # But that may be expensive; for now, implement as einsum
                    
                    # Build label tensor representation
                    # Map each axis position to a label index
                    # For the label remapping, we need to assign a label per axis value

                    # Generate label matrices for the output tensor shape
                    def create_label_array(axis: int, label_char: str):
                        # Array of shape shape with True at positions where index == label_char
                        lbl_arr = (axis_indices[axis] == label_char).astype(jnp.float32)
                        return lbl_arr

                    # Build label arrays for each axis
                    label_arrays = {}
                    for axis, label_char in c_mapping.items():
                        label_arrays[label_char] = create_label_array(axis, label_char)

                    # For sum over indices with matching labels, construct a mask
                    masks = list(label_arrays.values())

                    # Combine masks via logical AND over all labels
                    combined_mask = functools.reduce(lambda a, b: a & b, masks)
                    # Sum over all indices where combined_mask is True
                    # To perform the sum, multiply tensor by mask (broadcasted)
                    tensor_masked = tensor * combined_mask
                    # Sum over all indices
                    result = jnp.sum(tensor_masked)
                    # Reshape result to output tensor shape if needed
                    # For simplicity, return scalar if sum reduces to scalar
                    # Alternatively, broadcast result over output shape
                    # But since the sum reduces to scalar, return as shape scalar
                    return result
                return operation
            # Append ArrayOperation wrapping the operation
            array_op = ArrayOperation(make_operation())
            basis_list.append(array_op)
        return basis_list

    def _assign_characters_to_partition(self, partition: List[List[int]]) -> Dict[int, str]:
        """
        Assign unique characters to each subset in a partition.
        """
        label_dict: Dict[int, str] = {}
        for subset, char in zip(partition, CHARACTERS):
            for axis in subset:
                label_dict[axis] = char
        return label_dict

    def _enumerate_partitions(self, axes: List[int]) -> List[List[List[int]]]:
        """
        Enumerate all valid partitions of the set of axes.
        Only partitions where axes that are permuted together are grouped.
        For simplicity, generate all partitions (Bell number), then filter
        based on symmetry constraints. Here, we assume all axes are permutable.
        """
        # Generate all set partitions
        # Use standard set partition enumeration
        return list(self._set_partitions(axes))

    def _set_partitions(self, set_: List[int]) -> List[List[List[int]]]:
        """
        Recursively generate set partitions.
        """
        if not set_:
            return [[]]
        first = set_[0]
        rest = set_[1:]
        partitions = []
        for smaller in self._set_partitions(rest):
            # Insert first into existing subsets
            for i, subset in enumerate(smaller):
                new_partition = [s.copy() for s in smaller]
                new_partition[i].append(first)
                partitions.append(new_partition)
            # Or put first as a new subset
            new_partition = [[first]] + [s.copy() for s in smaller]
            partitions.append(new_partition)
        return partitions

```

## dataset_loader.py

```python
## dataset_loader.py
import os
import numpy as np
import jax
import jax.numpy as jnp
import tensorflow_datasets as tfds
from typing import Tuple, Optional
from collections import namedtuple

# Define Dataset structure
Dataset = namedtuple('Dataset', ['inputs', 'labels'])

def load_dataset(
    name: str,
    split: str,
    seed: int = 42,
    image_size: int = 28,
    flatten: bool = True
) -> Dataset:
    """
    Load dataset specified by 'name' with a given split. Preprocess according
    to dataset type (images or text). Returns Dataset with 'inputs' and 'labels'.
    """
    # Map dataset names to loader functions
    if name.lower() == 'fashionmnist':
        ds = tfds.load('fashion_mnist', split=split, as_supervised=True, shuffle_files=True, shuffle_seed=seed)
        inputs, labels = _process_image_dataset(ds, image_size, flatten)
        return Dataset(inputs=inputs, labels=labels)
    elif name.lower() == 'cifar10':
        ds = tfds.load('cifar10', split=split, as_supervised=True, shuffle_files=True, shuffle_seed=seed)
        inputs, labels = _process_image_dataset(ds, image_size, flatten)
        return Dataset(inputs=inputs, labels=labels)
    elif name.lower() == 'lm1b':
        # For LM1B, assume dataset stored as text file or tfds dataset
        return _load_lm1b_dataset(split, seed)
    else:
        raise ValueError(f"Unsupported dataset name: {name}")

def _process_image_dataset(ds, image_size: int, flatten: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert image dataset to array, normalize, resize, and optionally flatten.
    """
    images = []
    labels = []
    for img, lbl in tfds.as_numpy(ds):
        img = _preprocess_image(img, image_size)
        images.append(img)
        labels.append(lbl)
    images = np.stack(images, axis=0)
    labels = np.array(labels)
    # Convert to jax arrays
    inputs = jnp.array(images)
    labels = jnp.array(labels)
    return inputs, labels

def _preprocess_image(image: np.ndarray, size: int) -> np.ndarray:
    """
    Resize (if necessary), normalize pixel values to [0,1].
    """
    # If image is not the target size, resize may be needed here.
    # For simplicity, assume datasets are already appropriately sized.
    image = image.astype(np.float32) / 255.0
    if image.shape[-1] != 1 and len(image.shape) == 3:
        # ensure shape is (H, W, C)
        pass
    if image.shape[0] != size or image.shape[1] != size:
        # Resize to target size
        # Use simple resize via jax.image.resize or numpy
        import cv2
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    if len(image.shape) == 2:
        # Add channel dimension if grayscale
        image = np.expand_dims(image, axis=-1)
    return image

def _load_lm1b_dataset(split: str, seed: int) -> Dataset:
    """
    Load or generate LM1B dataset (character or token sequences).
    This is a placeholder: requires external dataset access.
    """
    # For demonstration, generate synthetic data
    # Let's assume sequences are integer token IDs in [0, 10000]
    num_samples = 10000
    sequence_length = 16
    rng = np.random.RandomState(seed)
    inputs = rng.randint(0, 10000, size=(num_samples, sequence_length))
    labels = rng.randint(0, 10000, size=(num_samples, sequence_length))
    # Convert to jax arrays
    inputs = jnp.array(inputs)
    labels = jnp.array(labels)
    return Dataset(inputs=inputs, labels=labels)

def load_training_data(seed: int = 42, image_size: int = 28, flatten: bool = True) -> Dataset:
    """
    Load and process training dataset based on the configuration.
    """
    # Get dataset name from global config or hardcoded here
    # For demonstration, let's assume 'FashionMNIST'
    dataset_name = 'FashionMNIST'  # Or retrieve from a config
    # Use 'train' split
    return load_dataset(dataset_name, 'train', seed=seed, image_size=image_size, flatten=flatten)

def load_validation_data(seed: int = 42, image_size: int = 28, flatten: bool = True) -> Dataset:
    """
    Load and process validation dataset.
    """
    dataset_name = 'FashionMNIST'
    # Use 'validation' or split from 'train' split
    # Since TFDS splits don't always have validation, split manually
    # For simplicity, load full train and split
    full_ds = tfds.load('fashion_mnist', split='train', as_supervised=True, shuffle_files=True, shuffle_seed=seed)
    total_size = tfds.as_numpy(full_ds).datum_size if hasattr(full_ds, 'datum_size') else 60000
    val_size = int(0.1 * total_size)
    # Shuffle with seed
    ds_shuffled = tfds.load('fashion_mnist', split='train', as_supervised=True, shuffle_files=True, shuffle_seed=seed)
    ds_list = list(tfds.as_numpy(ds_shuffled))
    rng = np.random.RandomState(seed)
    rng.shuffle(ds_list)
    val_ds = ds_list[:val_size]
    inputs, labels = _process_image_list(val_ds, image_size, flatten)
    return Dataset(inputs=inputs, labels=labels)

def load_test_data(seed: int = 42, image_size: int = 28, flatten: bool = True) -> Dataset:
    """
    Load and process test dataset.
    """
    dataset_name = 'FashionMNIST'
    ds = tfds.load('fashion_mnist', split='test', as_supervised=True)
    inputs, labels = _process_image_dataset(ds, image_size, flatten)
    return Dataset(inputs=inputs, labels=labels)

def _process_image_list(ds_list, image_size: int, flatten: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Process list of (image, label) tuples into arrays.
    """
    images = []
    labels = []
    for img, lbl in ds_list:
        img = _preprocess_image(img, image_size)
        images.append(img)
        labels.append(lbl)
    images = np.stack(images, axis=0)
    labels = np.array(labels)
    inputs = jnp.array(images)
    labels = jnp.array(labels)
    return inputs, labels
```

## evaluation.py

```python
## evaluation.py
"""
This module provides functions to compute evaluation metrics during meta-training and final evaluation.
Metrics implemented:
- Classification accuracy (success rate)
- Rank correlation (Kendall's tau) for regression or prediction success ranking

It is designed to handle model predictions and true labels in NumPy or JAX device array formats.
"""

import numpy as np
from scipy.stats import kendalltau

def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict:
    """
    Compute evaluation metrics from model predictions and true labels.

    Args:
        predictions (np.ndarray): Model outputs.
            For classification: shape (batch_size, num_classes), e.g., logits or probabilities.
            For regression or ranking tasks: shape (batch_size,) scalar predictions.
        targets (np.ndarray): True labels.
            For classification: integer class labels, shape (batch_size,).
            For regression or ranking: real-valued scalar labels, shape (batch_size,).

    Returns:
        dict: Dictionary containing evaluation metrics:
            - 'accuracy': float, success rate for classification tasks.
            - 'kendall_tau': float, Kendall's rank correlation coefficient.
            - 'p_value': float, p-value associated with Kendall's tau.
    """
    metrics = {}

    # Convert inputs to numpy arrays if they are JAX DeviceArrays
    preds = np.array(predictions)
    truths = np.array(targets)

    # Determine if the task is classification or regression based on preds shape
    if preds.ndim > 1:
        # Assume classification if predictions are logits/probabilities
        predicted_labels = preds.argmax(axis=-1)
    else:
        # For scalar predictions, treat as regression or ranking
        predicted_labels = preds

    # Compute accuracy for classification
    correct_predictions = np.sum(predicted_labels == truths)
    accuracy = correct_predictions / len(truths)
    metrics['accuracy'] = accuracy

    # Compute Kendall's tau correlation coefficient
    # Flatten arrays in case of higher dimensions
    tau, p_value = kendalltau(preds.flatten(), truths.flatten())
    metrics['kendall_tau'] = tau
    metrics['p_value'] = p_value

    return metrics
```

## main.py

```python
# main.py
import os
import yaml
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from itertools import product
from dataset_loader import load_training_data, load_validation_data, load_test_data
from basis_generator import BasisGenerator
from model import DeepUNF
from trainer import MetaTrainer
from evaluation import compute_metrics

def main():
    # 1. Load configuration from 'config.yaml'
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Extract seed and device
    seed = config.get('sample', {}).get('seed', 42)
    device_type = config.get('sample', {}).get('device', 'gpu')  # 'gpu' or 'cpu'
    # Set JAX device configuration
    if device_type == 'cpu':
        jax.config.update('jax_platform_name', 'cpu')
    else:
        jax.config.update('jax_platform_name', 'gpu')
    # Set PRNG key for reproducibility
    rng = jax.random.PRNGKey(seed)

    # 2. Load architecture specification and create ArchitectureSpec object
    # Assume 'architecture' section exists and defines model_type, hidden_layers, etc.
    arch_conf = config.get('architecture', {})
    model_type = arch_conf.get('model_type', 'MLP')
    hidden_layers = arch_conf.get('hidden_layers', [600, 600, 600])
    activation = arch_conf.get('activation', 'relu')
    weight_spec_dict = arch_conf.get('weight_specification', {})  # Nested dict describing weights

    # Build ArchitectureSpec object (from your 'architecture_spec.py')
    # Placeholder: create actual ArchitectureSpec with shape info, symmetry info
    # For demonstration, we assume a function load_architecture_spec provides this
    # You should implement this function or load a pre-defined spec
    from architecture_spec import ArchitectureSpec, LayerSpec, TensorSpec, PermutationGroup
    architecture_spec = ArchitectureSpec(
        layers=[],  # fill in per actual architecture
        symmetry_groups={}  # fill in per weight spec
    )
    # In practice, parse 'weight_spec_dict' to fill these

    # 3. Instantiate BasisGenerator
    basis_gen = BasisGenerator(architecture_spec)

    # 4. Generate basis functions for each tensor in the architecture
    # Assume architecture_spec provides a list of all tensor specs
    tensor_specs: list[TensorSpec] = []
    for layer in architecture_spec.layers:
        tensor_specs.extend(architecture_spec.get_tensor_spec(layer.name))
    basis_per_tensor: Dict[str, List[object]] = {}
    for spec in tensor_specs:
        basis_functions = basis_gen.generate_basis(spec)
        basis_per_tensor[spec] = basis_functions

    # 5. Build the deep neural functional (UNF)
    # For simplicity, assume we construct a list of basis layers with nonlinearities
    # Stack multiple NeuralLayers in DeepUNF
    # For demonstration, create 3 layers with basis functions
    basis_layers = []
    for _ in range(3):
        # For each layer, assemble a list of basis functions
        # Here, we pick basis functions for each tensor; in real code, select relevant
        basis_layer = list(basis_per_tensor.values())[0]  # placeholder for each layer
        basis_layers.append(basis_layer)
    nonlinear_fn = jax.nn.relu
    pooling_fn = jnp.sum  # for invariance; replace as needed

    def unf_forward(inputs):
        # inputs: list of input tensors (e.g., weight tensors)
        unf = DeepUNF(basis_layers=basis_layers, nonlinear=nonlinear_fn, pooling=pooling_fn)
        return unf(inputs)

    unf_model = hk.transform(unf_forward)

    # 6. Load datasets based on architecture/model_type
    dataset_name = config.get('dataset', {}).get('name', 'FashionMNIST')
    if model_type == 'MLP':
        train_data = load_training_data(seed=seed)
        val_data = load_validation_data(seed=seed)
        test_data = load_test_data(seed=seed)
    elif model_type == 'CNN':
        train_data = load_training_data(seed=seed)
        val_data = load_validation_data(seed=seed)
        test_data = load_test_data(seed=seed)
        # Additional preprocessing for CNN if needed
    elif model_type in ('RNN', 'Transformer'):
        train_data = load_training_data(seed=seed)
        val_data = load_validation_data(seed=seed)
        test_data = load_test_data(seed=seed)
        # Additional for text datasets
    else:
        raise ValueError(f"Unsupported architecture/model_type: {model_type}")

    # 7. Initialize meta-parameters (coefficients of basis functions)
    total_basis_size = sum(len(basis_per_tensor[spec]) for spec in basis_per_tensor)
    init_coeffs = np.zeros(total_basis_size, dtype=np.float32)
    coeffs = jnp.array(init_coeffs)

    # Placeholder: Build a mapping from layers to basis sizes
    layer_basis_sizes = {}
    for spec in basis_per_tensor:
        layer_basis_sizes[str(spec)] = len(basis_per_tensor[spec])

    # 8. Setup the meta-optimizer (via Evolution Strategies)
    meta_optimizer_config = config['training'].get('optimizer', {})
    meta_lr = meta_optimizer_config.get('step_size', 0.001)
    population_size = config['training'].get('meta_population_size', 64)
    noise_std = config['training'].get('noise_std', 0.01)
    meta_iterations = config['training'].get('meta_iterations', 50000)
    inner_steps = config['training'].get('inner_steps', 2000)

    opt = optax.adam(meta_lr)
    opt_state = opt.init(coeffs)

    def apply_basis_layers(inputs, coeffs_array: jnp.ndarray):
        """
        Apply basis functions weighted by coefficients.
        """
        out = []
        idx = 0
        for spec in basis_per_tensor:
            basis_list = basis_per_tensor[spec]
            size = len(basis_list)
            coeffs_layer = coeffs_array[idx:idx + size]
            idx += size
            res = 0
            for c, basis_fn in zip(coeffs_layer, basis_list):
                # basis_fn: applies to relevant tensors
                out_fn = basis_fn(*inputs)
                res = res + c * out_fn
            out.append(res)
        return out

    # 9. Meta-training main loop
    for meta_iter in range(meta_iterations):
        # Sample noise for ES
        rng, noise_key = jax.random.split(rng)
        noise = jax.random.normal(noise_key, shape=(population_size, coeffs.shape[0]))
        # Generate noisy candidate coefficients
        coeffs_candidates = coeffs + noise * noise_std

        rewards = []

        # For each candidate in population
        for i in range(population_size):
            candidate_coeffs = coeffs_candidates[i]
            # Initialize target network weights; can be random or pre-trained
            target_weights = None  # Placeholder: define your target weights initialization
            # Perform inner training
            # For synthetic datasets, generate or load corresponding weights
            # Here, we assume a function init_target_weights() exists
            target_weights = None  # Placeholder
            # Since actual target weights depend on dataset and model,
            # implement architecture-specific weight initialization as needed.
            # For demonstration, skip actual training loop
            # Instead, simulate evaluation metric (e.g., assume success rate > 0.8)
            reward = 0.8  # Placeholder: replace with actual evaluation after inner training
            rewards.append(reward)

        rewards = np.array(rewards)
        # Normalize rewards
        rewards_mean = np.mean(rewards)
        rewards_std = np.std(rewards) + 1e-8
        norm_rewards = (rewards - rewards_mean) / rewards_std

        # Compute gradient estimate (REINFORCE-like)
        grad_estimate = (1. / (population_size * noise_std)) * np.dot(noise.reshape(population_size, -1).T, norm_rewards)

        # Update coefficients
        updates, opt_state = opt.update(grad_estimate, opt_state)
        coeffs = optax.apply_updates(coeffs, updates)

        # Log progress every 1000 steps
        if (meta_iter + 1) % 1000 == 0:
            print(f"Meta-iteration {meta_iter + 1}/{meta_iterations}")
            # Optionally, evaluate on validation set
            # For simplicity, skip inner train evaluation here

    # 10. Final evaluation with trained coefficients
    # Apply the deep UNF to actual weights of target network (or synthetic data)
    # For demonstration, generate dummy weight tensors matching architecture spec
    dummy_inputs = []
    for spec in basis_per_tensor:
        shape = spec.shape  # placeholder: real shape from spec
        dummy_tensor = jnp.zeros(shape)
        dummy_inputs.append(dummy_tensor)

    unf_apply = hk.without_apply_rng(hk.transform(lambda inputs: DeepUNF(basis_layers=basis_layers, nonlinear=jax.nn.relu, pooling=jnp.sum)(inputs)))
    final_output = unf_apply.apply({}, dummy_inputs)

    # Save or log the trained coefficients and model as needed
    # For example:
    print("Meta-training completed.")
    print("Final basis coefficients:", coeffs)

if __name__ == "__main__":
    main()
```

## model.py

```python
## model.py
import jax
import jax.numpy as jnp
import haiku as hk
from typing import List, Callable, Optional
from basis_generator import ArrayOperation  # Import ArrayOperation for basis application

class NeuralLayer(hk.Module):
    """
    Represents a single basis function layer in the neural functional.
    Applies a learned linear combination over the basis functions (ArrayOperations),
    taking input tensors and producing an output tensor with the same shape.
    """
    def __init__(
        self,
        basis_functions: List[ArrayOperation],
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.basis_functions = basis_functions
        self.num_basis = len(basis_functions)
        # Initialize learned coefficients for basis functions
        # Coefficients are parameters to be optimized
        self.coefficients = hk.get_parameter(
            "coefficients",
            shape=[self.num_basis],
            init=hk.initializers.RandomNormal(stddev=0.1)
        )

    def __call__(self, input_tensors: List[jnp.ndarray]) -> jnp.ndarray:
        """
        Apply a linear combination of basis functions to input tensor(s),
        sum over basis with learned coefficients.
        Expect input_tensors as a list of tensors matching basis input expectations.
        """
        # Apply each basis function to the input tensors
        basis_outputs = []
        for basis_fn in self.basis_functions:
            # Apply basis function, which is a ArrayOperation
            out = basis_fn(*input_tensors)  # Each basis fn takes multiple tensors if needed
            basis_outputs.append(out)
        # Stack basis outputs into array of shape (num_basis, shape...)
        basis_stack = jnp.stack(basis_outputs, axis=0)  # shape: (B, ...)

        # Use coefficients to form a weighted sum
        # Expand coefficients to match basis array
        coeffs = jnp.reshape(self.coefficients, [self.num_basis] + [1] * (basis_stack.ndim -1))
        # weighted sum over basis functions
        combined = jnp.sum(coeffs * basis_stack, axis=0)
        return combined

class DeepUNF(hk.Module):
    """
    Stacks multiple NeuralLayers with nonlinear activations to form the deep universal neural functional.
    Optionally performs pooling for invariance after the last layer.
    """
    def __init__(
        self,
        basis_layers: List[List[ArrayOperation]],
        nonlinear: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
        pooling: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        name: Optional[str] = None
    ):
        """
        Args:
            basis_layers: List of list of basis functions (ArrayOperation) for each layer.
            nonlinear: Activation function to apply after each layer (default ReLU).
            pooling: Optional global pooling function for invariance (e.g., sum, mean).
        """
        super().__init__(name=name)
        self.basis_layers = basis_layers
        self.nonlinear = nonlinear
        self.pooling = pooling
        # Initialize list of neural layers
        self.layers = [
            NeuralLayer(basis_fn_list, name=f"neural_layer_{i}")
            for i, basis_fn_list in enumerate(self.basis_layers)
        ]

    def __call__(self, input_tensors: List[jnp.ndarray]) -> jnp.ndarray:
        """
        Forward pass through stacked neural layers with nonlinearities.
        input_tensors: list of tensors that serve as initial input (e.g., weight tensors)
        Returns pooled invariant or equivariant output tensor.
        """
        current_repr = input_tensors
        for layer in self.layers:
            # Apply a NeuralLayer
            current_repr = layer(current_repr)
            # Apply nonlinearity
            current_repr = self.nonlinear(current_repr)

        # Optionally apply pooling for invariance
        if self.pooling is not None:
            # Pool over all spatial or feature dimensions as needed
            # For example, sum or mean over all axes except the batch
            # Here, assuming the batch dimension is 0
            pooled = self.pooling(current_repr, axis=tuple(range(1, current_repr.ndim)))
            return pooled
        else:
            # Return final representation if no pooling
            # Could be multiple tensors or a single tensor
            return current_repr
```

## trainer.py

```python
# trainer.py
import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import List, Dict, Any, Tuple, Callable, Optional
from jax import jit, grad, vmap, random
from evaluation import compute_metrics
from dataset_loader import load_training_data, load_validation_data, load_test_data
from basis_generator import BasisGenerator
from model import DeepUNF
import yaml

# Load configuration from 'config.yaml'
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Extract training settings
learning_rate = config['training'].get('learning_rate', 0.001)
batch_size = config['training'].get('batch_size', 128)
inner_steps = config['training'].get('inner_steps', 2000)
meta_iterations = config['training'].get('meta_iterations', 50000)
population_size = config['training'].get('meta_population_size', 64)
noise_std = config['training'].get('noise_std', 0.01)
meta_optimizer_config = config['training']['optimizer']
dataset_name = config['dataset'].get('name', 'FashionMNIST')
sample_seed = config['sample'].get('seed', 42)

# Load datasets
train_dataset = load_training_data(seed=sample_seed)
val_dataset = load_validation_data(seed=sample_seed)
test_dataset = load_test_data(seed=sample_seed)

# Define dataset iterator for inner training
def dataset_iterator(dataset, batch_size, rng):
    # yields batches indefinitely
    n = len(dataset.inputs)
    perm = jax.random.permutation(rng, n)
    while True:
        for i in range(0, n, batch_size):
            indices = perm[i: i + batch_size]
            yield dataset.inputs[indices], dataset.labels[indices]

# Instantiate basis generator with architecture spec (assuming loaded or predefined)
# For demonstration, create a placeholder architecture spec
architecture_spec = ...  # Placeholder: should be loaded/predefined for your architecture
basis_generator = BasisGenerator(architecture_spec)

# Generate basis functions for model layers (assuming list of tensor specs)
# For simplicity, suppose architecture_spec provides tensor specs
tensor_specs = ...  # List of TensorSpec objects for all weight tensors
basis_per_layer = {}
for tensor_spec in tensor_specs:
    basis_per_layer[tensor_spec] = basis_generator.generate_basis(tensor_spec)

# Initialize meta-parameters: coefficients for each basis in each layer
# structure: dict[layer_name] -> parameter array shape: [number_of_basis]
# For simplicity, pack all coefficients in a single vector
# Suppose we flatten all basis coefficients across layers into a vector
def initialize_coefficients():
    coeffs_list = []
    layer_basis_sizes = {}
    for layer_name, basis_list in basis_per_layer.items():
        size = len(basis_list)
        layer_basis_sizes[layer_name] = size
        # initialize with zeros or small random (here zeros)
        coeffs_list.append(np.zeros(size, dtype=np.float32))
    # concatenate all coefficients
    return np.concatenate(coeffs_list), layer_basis_sizes

# Create optimizer for meta-parameters (coefficients)
meta_optimizer = optax.adam(learning_rate=1e-3)
params = initialize_coefficients()
opt_state = meta_optimizer.init(params)

# Helper to reconstruct per-layer basis coefficient slices
def unpack_coeffs(params_flat, layer_basis_sizes):
    slices = {}
    idx = 0
    for layer_name, size in layer_basis_sizes.items():
        slices[layer_name] = params_flat[idx: idx + size]
        idx += size
    return slices

# Placeholder for building target model (architecture specific)
def init_target_weights(rng_key):
    # Returns initial weights of target model, as a nested dict or list
    # Replace with your architecture's initialization
    pass

@jit
def apply_f(inputs: List[jnp.ndarray], params_coeffs: jnp.ndarray, layer_basis_sizes: Dict[str, int], basis_per_layer: Dict):
    """
    Apply the neural functional f to compute an update.
    - inputs: list of tensors (layer weight tensors)
    - params_coeffs: array of concatenated basis coefficients
    - layer_basis_sizes: dict of size per layer
    - basis_per_layer: dict of layer_name -> list of basis ArrayOperation
    """
    start_idx = 0
    result_updates = []
    for layer_name, basis_list in basis_per_layer.items():
        size = layer_basis_sizes[layer_name]
        coeffs = params_coeffs[start_idx: start_idx + size]
        start_idx += size
        # Compute linear combo of basis functions
        # basis_list: List[ArrayOperation], coeffs: shape=[size]
        # Apply each basis function to appropriate input tensors
        layer_update = 0
        for c, basis_fn in zip(coeffs, basis_list):
            out = basis_fn(*inputs)  # assuming basis_fn takes input tensors
            layer_update += c * out
        result_updates.append(layer_update)
    return result_updates

# Inner training loop
def inner_train(rng_key, init_weights, coeffs, layer_basis_sizes, basis_per_layer, train_iter, dataset_iter):
    """
    Performs inner-loop training for a target network with optimizer parametrized by current coeffs.
    """
    weights = init_weights
    for step in range(inner_steps):
        batch_inputs, batch_labels = next(dataset_iter)
        # Compute gradients w.r.t. weights
        # Here, assume loss_fn is defined for target model, depends on architecture
        def loss_fn(w):
            preds = target_model_apply(w, batch_inputs)  # architecture-specific
            loss = ...  # e.g., cross-entropy
            return loss

        grads = grad(loss_fn)(weights)

        # Compute updates via the current UNF - based on basis functions
        # Inputs to basis functions: need to be extracted from weights
        # Extract relevant tensors for basis (depends on architecture)
        basis_inputs = ...  # depends on architecture
        # Compute update
        updates = apply_f(basis_inputs, coeffs, layer_basis_sizes, basis_per_layer)
        # Update weights (assuming simple SGD)
        # Update rule depends on optimizer, assuming explicit SGD here:
        weights = {k: weights[k] - learning_rate * grads[k] + updates[k] for k in weights}
    # After inner steps, evaluate final performance
    final_loss = loss_fn(weights)
    final_acc = ...  # compute accuracy on validation set
    return final_loss, final_acc

# Vectorized evaluation over population noise
@jit
def evaluate_population(rng, base_coeffs, noise_vectors, basis_per_layer, layer_basis_sizes, dataset_iter):
    """
    Evaluate each noisy candidate in the population.
    Return array of rewards (performance metrics).
    """
    def evaluate_one(rng, noise):
        coeffs_noisy = base_coeffs + noise * noise_std
        init_weights = init_target_weights(rng)
        val_loss, val_acc = inner_train(rng, init_weights, coeffs_noisy, layer_basis_sizes, basis_per_layer, inner_steps, dataset_iter)
        # Use validation accuracy or negative loss as reward
        reward = val_acc  # or -val_loss if optimizing loss
        return reward

    rngs = jax.random.split(rng, len(noise_vectors))
    rewards = vmap(evaluate_one)(rngs, noise_vectors)
    return rewards

# Meta-optimization step
def meta_update(rng_key, current_params, meta_opt_state, basis_per_layer, layer_basis_sizes, dataset_iter):
    """
    Perform one meta-iteration with ES.
    """
    # Sample noise vectors
    key, subkey = jax.random.split(rng_key)
    noise_shape = (population_size, params.shape[0])
    noise_vectors = jax.random.normal(subkey, shape=noise_shape)

    # Evaluate population
    rewards = evaluate_population(key, current_params, noise_vectors, basis_per_layer, layer_basis_sizes, dataset_iter)

    # Compute gradient estimation via ES
    normalized_rewards = (rewards - jnp.mean(rewards)) / (jnp.std(rewards) + 1e-8)
    # Gradient estimate: weighted sum
    grad_estimate = (1. / (population_size * noise_std)) * jnp.mean(
        (noise_vectors.T * normalized_rewards).T, axis=0
    )

    # Meta optimizer update
    updates, new_opt_state = meta_optimizer.update(grad_estimate, meta_opt_state)
    new_params = optax.apply_updates(current_params, updates)
    return new_params, new_opt_state

# Main meta-training loop
def meta_train(rng_seed: int = 42):
    rng = jax.random.PRNGKey(rng_seed)
    dataset_iter = dataset_iterator(train_dataset, batch_size, rng)
    coeffs, layer_basis_sizes = initialize_coefficients()

    # Convert coefficients to jax DeviceArray
    coeffs = jnp.array(coeffs)

    for meta_iter in range(meta_iterations):
        rng, subkey = jax.random.split(rng)
        coeffs, opt_state = meta_update(subkey, coeffs, opt_state, basis_per_layer, layer_basis_sizes, dataset_iter)

        # Optional: evaluate current f periodically
        if meta_iter % 1000 == 0:
            print(f"Meta iteration {meta_iter}")
            # Evaluate on validation set
            # For simplicity, perform a single inner train with current coeffs
            init_w = init_target_weights(rng)
            val_loss, val_acc = inner_train(rng, init_w, coeffs, layer_basis_sizes, basis_per_layer, inner_steps, dataset_iter)
            print(f"Validation accuracy: {val_acc:.4f}")

        # Save checkpoint if needed

    return coeffs

# Run meta-training
if __name__ == "__main__":
    final_coeffs = meta_train(rng_seed=sample_seed)

    # Final evaluation on test set
    init_w = init_target_weights(jax.random.PRNGKey(sample_seed))
    val_loss, val_acc = inner_train(jax.random.PRNGKey(sample_seed), init_w, final_coeffs, basis_per_layer, basis_per_layer, inner_steps, dataset_iterator(test_dataset, batch_size, jax.random.PRNGKey(sample_seed)))
    print(f"Final validation accuracy: {val_acc:.4f}")
```

---

**Note:** Generated code files are available in `C:\Projects\473-Capstone-Project-EP2C\Backend\example_driver\..\..\dataset_out\paper2code\universal_neural_functional\universal_neural_functional_repo`
