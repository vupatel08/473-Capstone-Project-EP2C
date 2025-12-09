## discrimination_module.py
import torch
import torch.nn.functional as F

class DiscriminationModule:
    """
    Purpose:
        This module computes class-level representations
        of real and synthetic graphs and calculates a discrimination
        loss based on cosine similarity to promote class-wise feature alignment.
    Dependencies:
        Uses torch for tensor operations and cosine similarity.
    """
    def __init__(self,
                 real_labels: torch.Tensor,
                 real_features: torch.Tensor,
                 synthetic_labels: torch.Tensor,
                 synthetic_features: torch.Tensor):
        """
        Initializes the module with node labels and features for real and synthetic graphs.

        Args:
            real_labels (torch.Tensor): shape (N,), class labels for real graph nodes.
            real_features (torch.Tensor): shape (N, d), node features for real graph.
            synthetic_labels (torch.Tensor): shape (N',), class labels for synthetic graph nodes.
            synthetic_features (torch.Tensor): shape (N', d), node features for synthetic graph.
        """
        self.real_labels = real_labels
        self.real_features = real_features
        self.synthetic_labels = synthetic_labels
        self.synthetic_features = synthetic_features

        # Determine number of classes
        self.num_classes = int(torch.max(torch.cat([real_labels, synthetic_labels])) + 1)

        # Internal placeholders for class-wise representations
        self.H = None  # real class representations
        self.H_prime = None  # synthetic class representations

    def compute_class_representations(self):
        """
        Compute class-level centroid vectors for real and synthetic graphs.
        These vectors are the mean of node features per class, incorporating neighborhood information.
        """
        device = self.real_features.device

        # Initialize tensors
        H = torch.zeros((self.num_classes, self.real_features.shape[1]), device=device)
        H_prime = torch.zeros((self.num_classes, self.synthetic_features.shape[1]), device=device)

        for c in range(self.num_classes):
            # Mask for class c in real graph
            real_mask = (self.real_labels == c)
            # Mask for class c in synthetic graph
            synth_mask = (self.synthetic_labels == c)

            if real_mask.sum() > 0:
                # Aggregate features of class c in real graph
                class_feats_real = self.real_features[real_mask]
                # Optionally, incorporate neighborhood info: here, just use raw features
                H[c] = class_feats_real.mean(dim=0)
            else:
                # If no node in class c, keep zero vector
                H[c] = torch.zeros(self.real_features.shape[1], device=device)

            if synth_mask.sum() > 0:
                # Similarly, for synthetic graph
                class_feats_synth = self.synthetic_features[synth_mask]
                H_prime[c] = class_feats_synth.mean(dim=0)
            else:
                H_prime[c] = torch.zeros(self.synthetic_features.shape[1], device=device)

        # Normalize class-wise vectors to unit norm for cosine similarity stability
        H_normalized = F.normalize(H, p=2, dim=1)
        H_prime_normalized = F.normalize(H_prime, p=2, dim=1)

        self.H = H_normalized
        self.H_prime = H_prime_normalized

    def discrimination_loss(self):
        """
        Compute the class-wise discrimination loss based on cosine similarity.
        The loss encourages high similarity for correct class pairs
        and dissimilarity for different class pairs.
        """
        if self.H is None or self.H_prime is None:
            raise RuntimeError("Call compute_class_representations() before discrimination_loss().")

        total_loss = 0.0
        C = self.num_classes

        for c in range(C):
            # Similarity for same class
            sim_same = torch.dot(self.H[c], self.H_prime[c])
            loss_same = 1 - sim_same  # We want to maximize similarity => minimize 1 - cosine_similarity

            # Dissimilarity for different classes
            for c2 in range(C):
                if c2 != c:
                    sim_diff = torch.dot(self.H[c], self.H_prime[c2])
                    loss_diff = sim_diff  # Want to minimize similarity, so add similarity term
                    total_loss += loss_diff

            total_loss += loss_same

        # Average over number of classes
        loss_value = total_loss / (C + C * (C -1))
        return loss_value

