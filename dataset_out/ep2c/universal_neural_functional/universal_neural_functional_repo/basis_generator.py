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

