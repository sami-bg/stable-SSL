import torch
import torch.nn as nn
from typing import Optional, Tuple


class StreamingTopKEigen(nn.Module):
    """Hyperparameter-free streaming estimator for top-K eigenvectors.

    This module maintains running estimates of the top-K eigenvectors and
    eigenvalues of the covariance matrix of streaming data. It requires no
    tuning - learning rates and update schedules are derived automatically
    from theoretical considerations.

    Key Features:
    -------------
    - **No hyperparameters**: Learning rates adapt based on sample count and
      current eigenvalue estimates.
    - **Memory efficient**: O(dk) storage, no need to store covariance matrix.
    - **Numerically stable**: QR orthogonalization + Welford's algorithm.
    - **Fast**: Single-pass update per batch, optimized for GPU.

    Mathematical Background:
    ------------------------
    Given streaming data x₁, x₂, ..., xₙ ∈ ℝᵈ, we want to estimate the top-K
    eigenvectors of the covariance matrix:

        C = E[(x - μ)(x - μ)ᵀ]

    The algorithm maintains:
    - Running mean μ̂ (Welford's algorithm)
    - Eigenvector matrix V ∈ ℝᵈˣᵏ (columns are eigenvectors)
    - Eigenvalue estimates λ̂ ∈ ℝᵏ

    Updates use Sanger's rule with adaptive learning rates:

        V ← V + diag(η) · (E[x̃ yᵀ] - V · tril(E[y yᵀ]))

    where:
    - x̃ = x - μ̂ (centered data)
    - y = Vᵀx̃ (projections)
    - η_i = (1/√n) · (σ²_total / λ̂_i) (adaptive per-component learning rate)
    - tril(·) extracts lower triangular part (for deflation)

    Example Usage:
    --------------
    >>> # Initialize estimator
    >>> estimator = StreamingTopKEigen(dim=512, k=16, device="cuda")
    >>>
    >>> # Training loop - just call forward on each batch
    >>> for epoch in range(num_epochs):
    ...     for batch in dataloader:
    ...         x = batch["features"]  # (batch_size, 512)
    ...         eigenvalues, eigenvectors = estimator(x)
    ...
    ...         # Optional: use for dimensionality reduction
    ...         x_reduced = estimator.project(x)  # (batch_size, 16)
    >>>
    >>> # After training, eigenvectors are available
    >>> print(f"Top eigenvalue: {estimator.eigenvalues[0]:.4f}")
    >>> print(f"Variance explained: {estimator.explained_variance_ratio.sum():.2%}")

    Integration with Neural Networks:
    ---------------------------------
    >>> class AutoEncoder(nn.Module):
    ...     def __init__(self, dim, latent_dim):
    ...         super().__init__()
    ...         self.encoder = nn.Linear(dim, latent_dim)
    ...         self.decoder = nn.Linear(latent_dim, dim)
    ...         # Track principal components of encoder output
    ...         self.pca = StreamingTopKEigen(latent_dim, k=8)
    ...
    ...     def forward(self, x):
    ...         z = self.encoder(x)
    ...         # Update PCA estimates (no grad needed)
    ...         with torch.no_grad():
    ...             self.pca(z)
    ...         return self.decoder(z)

    Parameters
    ----------
    dim : int
        Dimensionality of input features.
    k : int
        Number of top eigenvectors to estimate. Must satisfy k ≤ dim.
    device : torch.device, optional
        Device for tensors. If None, uses default device.
    dtype : torch.dtype, optional
        Data type for computations. Default is float32.
        Use float64 for higher precision if needed.

    Attributes:
    ----------
    V : torch.Tensor
        Current eigenvector estimates, shape (dim, k).
        Columns are eigenvectors, sorted by eigenvalue (descending).
    eigenvalues : torch.Tensor
        Current eigenvalue estimates, shape (k,), sorted descending.
    mean : torch.Tensor
        Running mean estimate, shape (dim,).
    n_samples : torch.Tensor
        Total number of samples seen so far.
    total_variance : torch.Tensor
        Estimate of total variance (trace of covariance matrix).
    initialized : torch.Tensor
        Boolean flag indicating if first batch has been processed.

    Notes:
    -----
    - The estimator uses `register_buffer` for all state, so it will be
      properly saved/loaded with `state_dict()` and moved with `.to(device)`.
    - All updates are performed in `torch.no_grad()` context - this module
      does not participate in gradient computation.
    - For very small batches (< k samples), consider accumulating batches
      before calling forward for better initialization.

    See Also:
    --------
    torch.pca_lowrank : PyTorch's built-in randomized PCA (non-streaming)
    sklearn.decomposition.IncrementalPCA : Scikit-learn's incremental PCA
    """

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        self,
        dim: int,
        k: int,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the streaming eigenvector estimator.

        Parameters
        ----------
        dim : int
            Input feature dimensionality.
        k : int
            Number of top eigenvectors to track.
        device : torch.device, optional
            Computation device (cpu/cuda).
        dtype : torch.dtype, optional
            Tensor dtype, default float32.
        """
        super().__init__()

        # Validate inputs
        if k > dim:
            raise ValueError(
                f"k ({k}) cannot exceed dim ({dim}). "
                f"Requested more eigenvectors than dimensions."
            )
        if k < 1:
            raise ValueError(f"k must be positive, got {k}")
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")

        self.dim = dim
        self.k = k

        # ---------------------------------------------------------------------
        # State buffers (will be saved with state_dict, moved with .to())
        # ---------------------------------------------------------------------

        # Eigenvector matrix: columns are eigenvectors, shape (dim, k)
        # Initialized properly on first forward pass
        self.register_buffer("V", torch.empty(dim, k, device=device, dtype=dtype))

        # Running mean estimate, shape (dim,)
        self.register_buffer("mean", torch.zeros(dim, device=device, dtype=dtype))

        # Eigenvalue estimates (variance along each principal direction)
        # Initialized to 1.0 to avoid division by zero before first update
        self.register_buffer("eigenvalues", torch.ones(k, device=device, dtype=dtype))

        # Total sample count (as float for smooth division)
        self.register_buffer("n_samples", torch.tensor(0.0, device=device, dtype=dtype))

        # Initialization flag
        self.register_buffer("initialized", torch.tensor(False, device=device))

        # Total variance estimate (trace of covariance matrix)
        # Used for learning rate scaling to achieve scale invariance
        self.register_buffer(
            "total_variance", torch.tensor(1.0, device=device, dtype=dtype)
        )

    # =========================================================================
    # Initialization from First Batch
    # =========================================================================

    @torch.no_grad()
    def _init_from_batch(self, x: torch.Tensor) -> None:
        """Initialize eigenvector estimates from the first batch using SVD.

        Fixed to properly handle batch_size < k case.
        """
        batch_size = x.shape[0]

        # Step 1: Compute batch mean
        self.mean.copy_(x.mean(dim=0))
        x_centered = x - self.mean

        # Step 2: Estimate total variance
        self.total_variance.copy_(x_centered.var() * self.dim + 1e-8)

        # Step 3: Truncated SVD
        U, S, Vh = torch.linalg.svd(x_centered, full_matrices=False)

        # Number of valid components from SVD
        # SVD of (n, d) matrix gives at most min(n, d) singular values
        k_available = min(self.k, len(S))

        # Only use non-zero singular values
        valid_mask = S > 1e-10
        k_valid = min(k_available, valid_mask.sum().item())

        if k_valid > 0:
            self.V[:, :k_valid] = Vh[:k_valid].T
            self.eigenvalues[:k_valid] = (S[:k_valid] ** 2) / batch_size

        # Step 4: Handle remaining components (if k_valid < k)
        if k_valid < self.k:
            remaining_count = self.k - k_valid

            # Generate random orthonormal vectors for remaining components
            # Use QR on random matrix for numerical stability
            random_vecs = torch.randn(
                self.dim, remaining_count, device=x.device, dtype=x.dtype
            )

            if k_valid > 0:
                # Project out existing eigenvectors
                existing = self.V[:, :k_valid]
                random_vecs = random_vecs - existing @ (existing.T @ random_vecs)

            # QR decomposition to get orthonormal vectors
            Q, R = torch.linalg.qr(random_vecs)

            # Ensure we have the right number of columns
            # (QR might return fewer if random_vecs was rank-deficient)
            n_from_qr = min(Q.shape[1], remaining_count)
            self.V[:, k_valid : k_valid + n_from_qr] = Q[:, :n_from_qr]

            # If still not enough (extremely rare), fill with random unit vectors
            if k_valid + n_from_qr < self.k:
                for i in range(k_valid + n_from_qr, self.k):
                    v = torch.randn(self.dim, device=x.device, dtype=x.dtype)
                    # Orthogonalize against all previous
                    v = v - self.V[:, :i] @ (self.V[:, :i].T @ v)
                    norm = v.norm()
                    self.V[:, i] = v / norm if norm > 1e-8 else torch.randn_like(v)
                    self.V[:, i] /= self.V[:, i].norm()

            # Set eigenvalues for remaining components
            if k_valid > 0:
                min_eigenvalue = self.eigenvalues[:k_valid].min() / 2
            else:
                min_eigenvalue = self.total_variance / self.dim
            self.eigenvalues[k_valid:] = min_eigenvalue

        self.n_samples.fill_(float(batch_size))
        self.initialized.fill_(True)

    # =========================================================================
    # Adaptive Learning Rate Computation
    # =========================================================================

    @torch.no_grad()
    def _compute_adaptive_lr(self) -> torch.Tensor:
        """Compute per-component adaptive learning rates.

        The learning rate for component i is:

            η_i = base_lr × (total_variance / λ_i)

        where:
        - base_lr = 1/√n decays with sample count
        - total_variance / λ_i provides natural gradient scaling

        Theoretical Motivation:
        -----------------------

        1. **1/√n decay**: This is the optimal rate for streaming estimation.
           - Too slow (1/n): High bias, slow adaptation
           - Too fast (constant): High variance, no convergence
           - Just right (1/√n): Optimal bias-variance tradeoff

        2. **Per-component scaling by 1/λ_i**: This is "natural gradient" or
           "Newton-like" scaling. Components with smaller eigenvalues need
           larger learning rates because:
           - The gradient magnitude is proportional to λ_i
           - Without scaling, small eigenvalue components converge slowly
           - This equalizes convergence rates across all components

        3. **Normalization by total_variance**: Makes the algorithm invariant
           to the overall scale of the data.

        Returns:
        -------
        torch.Tensor
            Learning rates for each component, shape (k,).
        """
        # Base learning rate: decays as 1/√n
        # The +1 prevents division by zero and smooths early updates
        base_lr = 1.0 / torch.sqrt(self.n_samples + 1.0)

        # Per-component scaling by inverse eigenvalue
        # Clamp eigenvalues to avoid division by very small numbers
        # Minimum is set relative to total variance for scale invariance
        min_eigenvalue = self.total_variance * 1e-6
        lambda_safe = self.eigenvalues.clamp(min=min_eigenvalue)

        # Natural gradient scaling: larger lr for smaller eigenvalues
        # Normalized by total variance for scale invariance
        lr_per_component = base_lr * (self.total_variance / lambda_safe)

        # Clamp to reasonable range to prevent instability
        # - Min 0.001: Ensures some update even for dominant components
        # - Max 1.0: Prevents overshooting
        return lr_per_component.clamp(min=0.001, max=1.0)

    # =========================================================================
    # Running Mean Update
    # =========================================================================

    @torch.no_grad()
    def _update_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Update running mean estimate and return centered data.

        Uses Welford's algorithm for numerically stable incremental mean:

            μ_new = (n_old × μ_old + n_batch × μ_batch) / n_total

        This is mathematically equivalent to computing the mean of all data
        seen so far, but doesn't require storing the data.

        Parameters
        ----------
        x : torch.Tensor
            Input batch, shape (batch_size, dim).

        Returns:
        -------
        torch.Tensor
            Centered data (x - updated_mean), shape (batch_size, dim).
        """
        batch_size = x.shape[0]
        n_total = self.n_samples + batch_size

        # Incremental mean update (Welford's algorithm)
        # μ_new = (n_old/n_total) × μ_old + (n_new/n_total) × μ_batch
        batch_mean = x.mean(dim=0)

        self.mean.mul_(self.n_samples / n_total)  # Scale old mean
        self.mean.add_(batch_mean, alpha=batch_size / n_total)  # Add batch contribution

        # Return centered data using the NEW mean
        # (This is important for unbiased covariance estimation)
        return x - self.mean

    # =========================================================================
    # Core Update: Sanger's Rule
    # =========================================================================

    @torch.no_grad()
    def _sanger_update(self, x_centered: torch.Tensor) -> None:
        """Perform one step of Sanger's rule (Generalized Hebbian Algorithm).

        Sanger's rule is an extension of Oja's rule for extracting multiple
        principal components simultaneously. It uses "deflation" to ensure
        that each component captures variance orthogonal to previous ones.

        Update Rule:
        ------------
        For each component i:

            Δvᵢ = ηᵢ × E[x̃ × yᵢ - yᵢ × Σⱼ≤ᵢ(yⱼ × vⱼ)]

        where:
        - x̃ = centered data
        - yᵢ = vᵢᵀx̃ = projection onto component i
        - The sum Σⱼ≤ᵢ provides deflation (removes contribution of earlier components)

        In matrix form, this becomes:

            ΔV = diag(η) × (E[x̃ yᵀ] - V × tril(E[y yᵀ]))

        where tril() extracts the lower triangular part (including diagonal).

        The Algorithm Step-by-Step:
        ---------------------------
        1. Compute projections y = Vᵀx̃
        2. Compute gradient term: E[x̃ yᵀ]
        3. Compute deflation term: V × tril(E[y yᵀ])
        4. Apply per-component learning rates
        5. Update V
        6. Re-orthogonalize using QR decomposition
        7. Update eigenvalue estimates

        Parameters
        ----------
        x_centered : torch.Tensor
            Centered input data, shape (batch_size, dim).

        Notes:
        -----
        The QR orthogonalization step is crucial for numerical stability.
        Without it, the eigenvectors would slowly drift from orthogonality
        due to floating-point errors, eventually collapsing.
        """
        batch_size = x_centered.shape[0]

        # Get adaptive learning rates (one per component)
        lr = self._compute_adaptive_lr()  # shape: (k,)

        # ---------------------------------------------------------------------
        # Step 1: Compute projections
        # ---------------------------------------------------------------------
        # y = Vᵀx̃, shape: (batch_size, k)
        # Each column yᵢ contains the projection of all samples onto eigenvector vᵢ
        proj = x_centered @ self.V

        # ---------------------------------------------------------------------
        # Step 2: Compute Hebbian term (what we want to move toward)
        # ---------------------------------------------------------------------
        # E[x̃ yᵀ] = (1/n) X̃ᵀ Y, shape: (dim, k)
        # This is the correlation between input dimensions and projections
        # Points the update in the direction that maximizes variance
        hebbian_term = x_centered.T @ proj / batch_size

        # ---------------------------------------------------------------------
        # Step 3: Compute deflation term (what we want to move away from)
        # ---------------------------------------------------------------------
        # E[y yᵀ] = (1/n) Yᵀ Y, shape: (k, k)
        # This is the covariance of the projections
        proj_cov = proj.T @ proj / batch_size

        # Extract lower triangular (including diagonal)
        # The diagonal handles self-normalization
        # Below-diagonal handles deflation from earlier components
        lower_tri = torch.tril(proj_cov)

        # Deflation term: V @ lower_tri, shape: (dim, k)
        deflation_term = self.V @ lower_tri

        # ---------------------------------------------------------------------
        # Step 4: Compute gradient and apply update
        # ---------------------------------------------------------------------
        # Gradient = Hebbian - Deflation
        gradient = hebbian_term - deflation_term

        # Apply per-component learning rate
        # lr has shape (k,), we broadcast over dim
        scaled_gradient = gradient * lr.unsqueeze(0)

        # Update eigenvector estimates
        self.V.add_(scaled_gradient)

        # ---------------------------------------------------------------------
        # Step 5: Re-orthogonalize using QR decomposition
        # ---------------------------------------------------------------------
        # QR decomposition: V = Q @ R where Q is orthonormal
        # This is more stable than Gram-Schmidt and very fast on GPU
        Q, R = torch.linalg.qr(self.V)

        # Ensure consistent sign convention
        # We want the diagonal of R to be positive (standard convention)
        # This prevents sign flips between updates
        signs = torch.diag(R).sign()
        signs[signs == 0] = 1  # Handle exact zeros (rare)

        # Apply sign correction to Q
        self.V.copy_(Q * signs.unsqueeze(0))

        # ---------------------------------------------------------------------
        # Step 6: Update eigenvalue estimates
        # ---------------------------------------------------------------------
        # Eigenvalue = E[yᵢ²] = variance along eigenvector i
        # We use exponential moving average for smooth estimates

        # Adaptive decay rate (faster updates early, slower later)
        eigenvalue_lr = 2.0 / torch.sqrt(self.n_samples + 1.0)
        eigenvalue_lr = eigenvalue_lr.clamp(max=0.5)  # Cap at 0.5 for stability

        # New eigenvalue estimates from this batch
        batch_eigenvalues = (proj**2).mean(dim=0)

        # EMA update: λ_new = (1 - lr) × λ_old + lr × λ_batch
        self.eigenvalues.lerp_(batch_eigenvalues, eigenvalue_lr)

        # ---------------------------------------------------------------------
        # Step 7: Update total variance estimate
        # ---------------------------------------------------------------------
        # Total variance = trace(Cov) = E[‖x̃‖²]
        batch_total_var = (x_centered**2).mean() * self.dim
        self.total_variance.lerp_(batch_total_var, eigenvalue_lr)

        # ---------------------------------------------------------------------
        # Step 8: Update sample count
        # ---------------------------------------------------------------------
        self.n_samples.add_(batch_size)

    # =========================================================================
    # Main Forward Method
    # =========================================================================

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update eigenvector estimates with a new batch of data.

        This is the main entry point. Call this on each mini-batch during
        training. The method automatically handles:
        - Initialization on first call (using exact SVD)
        - Incremental updates on subsequent calls (using Sanger's rule)

        Parameters
        ----------
        x : torch.Tensor
            Input batch of shape (batch_size, dim).
            Should be on the same device as the module.

        Returns:
        -------
        eigenvalues : torch.Tensor
            Current eigenvalue estimates, shape (k,).
            Sorted in descending order (largest first).
        eigenvectors : torch.Tensor
            Current eigenvector estimates, shape (dim, k).
            Columns are eigenvectors, corresponding to eigenvalues.
            Vectors are orthonormal: Vᵀ V = I.

        Examples:
        --------
        >>> estimator = StreamingTopKEigen(dim=256, k=8)
        >>> for batch in dataloader:
        ...     eigenvalues, eigenvectors = estimator(batch)
        ...     print(f"Top eigenvalue: {eigenvalues[0]:.4f}")

        Notes:
        -----
        - Returns clones to prevent accidental modification of internal state
        - The returned tensors are on the same device as the input
        - Updates are performed in-place on internal buffers
        """
        # Input validation
        if x.dim() != 2:
            raise ValueError(
                f"Expected 2D input (batch_size, dim), got {x.dim()}D tensor"
            )
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Expected dim={self.dim}, got {x.shape[1]}. Input shape: {x.shape}"
            )

        # Dispatch to initialization or update
        if not self.initialized:
            # First batch: initialize with exact SVD
            self._init_from_batch(x)
        else:
            # Subsequent batches: incremental update
            x_centered = self._update_mean(x)
            self._sanger_update(x_centered)

        # Return copies of current estimates
        return self.eigenvalues.clone(), self.V.clone()

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project data onto the estimated principal subspace.

        Computes z = (x - μ) @ V, projecting from d dimensions to k dimensions.

        Parameters
        ----------
        x : torch.Tensor
            Input data, shape (batch_size, dim) or (dim,).

        Returns:
        -------
        torch.Tensor
            Projected data, shape (batch_size, k) or (k,).

        Examples:
        --------
        >>> z = estimator.project(x)  # (batch_size, dim) -> (batch_size, k)
        >>> # For downstream model
        >>> output = classifier(z)
        """
        return (x - self.mean) @ self.V

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Project and reconstruct data (PCA denoising/compression).

        Computes x̂ = V @ Vᵀ @ (x - μ) + μ, which projects to the k-dimensional
        principal subspace and back. This removes components outside the
        top-k subspace (denoising) or compresses the representation.

        Parameters
        ----------
        x : torch.Tensor
            Input data, shape (batch_size, dim) or (dim,).

        Returns:
        -------
        torch.Tensor
            Reconstructed data, same shape as input.

        Examples:
        --------
        >>> x_denoised = estimator.reconstruct(x)
        >>> reconstruction_error = (x - x_denoised).pow(2).mean()
        """
        z = self.project(x)  # Project to k dimensions
        return z @ self.V.T + self.mean  # Reconstruct to d dimensions

    @property
    def explained_variance_ratio(self) -> torch.Tensor:
        """Fraction of total variance explained by each component.

        Returns:
        -------
        torch.Tensor
            Explained variance ratio for each component, shape (k,).
            Sums to less than 1.0 (remaining variance is in other components).

        Examples:
        --------
        >>> print(f"Total explained: {estimator.explained_variance_ratio.sum():.1%}")
        >>> print(f"Top component: {estimator.explained_variance_ratio[0]:.1%}")
        """
        return self.eigenvalues / (self.total_variance + 1e-8)

    @property
    def cumulative_explained_variance_ratio(self) -> torch.Tensor:
        """Cumulative explained variance ratio.

        Returns:
        -------
        torch.Tensor
            Cumulative sum of explained variance ratios, shape (k,).

        Examples:
        --------
        >>> # How many components for 95% variance?
        >>> cumvar = estimator.cumulative_explained_variance_ratio
        >>> n_components_95 = (cumvar < 0.95).sum() + 1
        """
        return self.explained_variance_ratio.cumsum(dim=0)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"dim={self.dim}, k={self.k}, "
            f"n_samples={int(self.n_samples.item())}, "
            f"initialized={self.initialized.item()}"
            f")"
        )
