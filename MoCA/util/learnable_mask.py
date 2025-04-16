import torch.nn as nn
import torch.nn.functional as F
import torch

import torch
import torch.nn as nn

class LearnableMask(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.mask_generator = nn.Linear(feature_dim, 1)

    def forward(self, x, mask_ratio):
        """
        x has shape [batch_size, num_patches, feature_dim].
        We want to keep K = (1 - mask_ratio) fraction of the patches
        based on the predicted importance scores.
        The returned mask has 1 for the masked patches and 0 for the unmasked patches.
        """
        # Compute importance scores (logits) and convert to probabilities
        logits = self.mask_generator(x).squeeze(-1)  # [B, N]
        probs = torch.softmax(logits, dim=1)         # [B, N]

        # Pick the top-K patches
        B, N, _ = x.shape
        K = int(N * (1 - mask_ratio))
        _, topk_indices = torch.topk(probs, K, dim=1, largest=True, sorted=True)  # [B, K]

        # Create mask: 1 for masked, 0 for unmasked
        hard_mask = torch.ones_like(probs)  # [B, N]
        hard_mask.scatter_(1, topk_indices, 0.0)

        # Straight-through version for the unmasked patches
        keep_mask = 1.0 - hard_mask
        st_keep_mask = keep_mask + (probs - keep_mask).detach()

        # Multiply x by the keep mask in the forward pass
        x_st = x * st_keep_mask.unsqueeze(-1)

        # Gather only the unmasked top-K patches to reduce dimension to [B, K, D]
        x_masked = torch.gather(
            x_st,
            dim=1,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        )

        # Construct ids_restore for the unmasked ordering
        ids_shuffle = torch.argsort(probs, dim=1, descending=True)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        return x_masked, hard_mask, ids_restore




def soft_topk(scores, k, temperature=1.0):
    """Soft approximation of top-k selection using a smooth sorting operator."""
    scores = scores / temperature  # Scale logits
    sorted_scores = torch.sort(scores, dim=1, descending=True)[0]  # Soft sort
    soft_topk = sorted_scores[:, :k]  # Select top-k (soft version)
    return soft_topk


def differentiable_argsort(values):
    """Approximate differentiable argsort using soft sorting."""
    soft_ranks = torch.argsort(values, dim=1)  # Sort indices
    return soft_ranks.float()  # Ensure gradients flow
