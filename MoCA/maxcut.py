import numpy as np
import torch

def compute_cut_value(W, x):
    """
    Compute the cut value for partition x.
    W is the weight matrix (symmetric 10x10 matrix).
    x is a numpy array of shape (10,) with entries +1 (masked) or -1 (unmasked).
    """
    return 0.5 * np.sum(W * (1 - np.outer(x, x)))

def mod_local_search(W, x_init, max_iter=1000, perturb_prob=0.1):
    """
    Improved Mod-Local Search algorithm for the max-cut problem with a fixed set size constraint.
    Only swaps between a masked node (x = +1) and an unmasked node (x = -1) are allowed.
    
    The improvement of a candidate swap (i, j) is computed as:
      Δ = G[i] + G[j] + 2*W[i,j],
    where G = (W @ x) * x.
    
    When a swap is performed, G is updated vectorized over all nodes.
    """
    n = len(x_init)
    x = x_init.copy()
    current_value = compute_cut_value(W, x)
    
    # Precompute the gain vector G = (W @ x) * x.
    G = (W @ x) * x

    for iteration in range(max_iter):
        # Find indices of masked and unmasked nodes.
        masked = np.where(x == 1)[0]
        unmasked = np.where(x == -1)[0]
        if masked.size == 0 or unmasked.size == 0:
            break

        # Compute candidate improvements for all pairs (i in masked, j in unmasked)
        cand_improvements = np.add.outer(G[masked], G[unmasked]) + 2 * W[np.ix_(masked, unmasked)]
        max_improv = cand_improvements.max()

        if max_improv > 1e-6:
            # Get indices corresponding to the maximum improvement.
            idx = np.unravel_index(np.argmax(cand_improvements), cand_improvements.shape)
            i_idx = masked[idx[0]]
            j_idx = unmasked[idx[1]]
            
            # Swap the selected nodes.
            x[i_idx] = -1
            x[j_idx] = 1
            current_value += max_improv
            
            # Save old gain values for the swapped nodes.
            old_G_i = G[i_idx]
            old_G_j = G[j_idx]
            
            # Update G vectorized for all nodes.
            G += 2 * x * (W[:, j_idx] - W[:, i_idx])
            
            # Correct the gain for the swapped nodes.
            G[i_idx] = -old_G_i
            G[j_idx] = -old_G_j
        else:
            # If no improving swap is found, perform a random swap with probability perturb_prob.
            if np.random.rand() < perturb_prob:
                masked_indices = np.where(x == 1)[0]
                unmasked_indices = np.where(x == -1)[0]
                if masked_indices.size and unmasked_indices.size:
                    i_idx = np.random.choice(masked_indices)
                    j_idx = np.random.choice(unmasked_indices)
                    x[i_idx] = -1
                    x[j_idx] = 1
                    current_value = compute_cut_value(W, x)
                    G = (W @ x) * x
            else:
                break  # Convergence: no improvement and no random swap.
    return x, current_value

def max_cut_mask(x, mask_ratio=0.75, max_iter=1000, perturb_prob=0.1):
    """
    x (tensor): [batch_size, num_p, window_size]
    mask_ratio (float): ratio of elements to be masked.
    
    For each sample, computes a max-cut partition on the absolute covariance matrix,
    with the fixed unmasked set size equal to num_p*(1-mask_ratio). Uses the improved
    mod_local_search function.
    """
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    x_masked = torch.zeros(N, L, device=x.device)
    
    for i in range(N):
        # Compute the absolute covariance matrix for sample i.
        #cov_matrix = torch.abs(torch.cov(x[i]))
        cov_matrix = torch.abs(fast_cov(x[i]))
        # Normalize the covariance matrix by its maximum absolute value.
        cov_matrix = cov_matrix / torch.max(cov_matrix)
        
        # Convert to numpy and set the diagonal to zero.
        W = cov_matrix.cpu().numpy()
        np.fill_diagonal(W, 0)
        
        # Create an initial partition: randomly choose len_keep nodes to be unmasked (-1).
        indices = np.random.permutation(L)
        x_init = np.ones(L, dtype=int)
        x_init[indices[:len_keep]] = -1

        # Run the improved mod_local_search algorithm.
        solution, _ = mod_local_search(W, x_init, max_iter=max_iter, perturb_prob=perturb_prob)
        x_masked[i] = torch.from_numpy(solution).to(x.device)
    
    # Generate indices for MAE-style masking.
    ids_shuffle = torch.argsort(x_masked, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return ids_keep, mask, ids_restore

def fast_cov(X):
    # X: [num_p, window_size]
    mean = X.mean(dim=1, keepdim=True)
    X_centered = X - mean
    cov = torch.mm(X_centered, X_centered.t()) / (X.shape[1] - 1)
    return cov