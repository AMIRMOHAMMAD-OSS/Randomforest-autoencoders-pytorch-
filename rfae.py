# rfae.py
# Copyright (c) 2025.
# MIT License.
"""
Random-Forest Autoencoder (RFAE) — PyTorch + scikit-learn implementation

This module implements the *Autoencoding Random Forests* method:
  • Encoder: diffusion-map embedding of the Random-Forest (RF) kernel
  • Out-of-sample encoding via Nyström
  • Decoder: k-NN in latent space with weighted averaging (continuous) and
    weighted majority vote (categorical). Optionally, an exact-but-slower
    variant can sample synthetic training inputs from the intersection of
    RF leaf regions (“maximum compatible rule”).

Reference:
  Vu, Kapar, Wright, Watson. *Autoencoding Random Forests.* NeurIPS 2025.
  arXiv:2505.21441 — https://arxiv.org/abs/2505.21441

Upstream R/Python reference repo (original authors):
  https://github.com/bips-hb/RFAE

Why this file:
  A clean, user-friendly Python API built on PyTorch to handle
  training, encoding, decoding, and evaluation end-to-end.

Key design choices
------------------
- RF training is delegated to scikit-learn (robust and fast).
- Kernel construction, spectral decomposition, and Nyström are implemented
  with NumPy/PyTorch (use GPU if available).
- Decoder defaults to the fast, accurate k-NN strategy recommended in the paper,
  with optional “exact” synthetic sampling of training inputs if you want strict
  adherence to the theoretical reconstruction pipeline.

Limitations
-----------
- Building the dense n×n kernel scales as O(n^2) memory. For n >> 10k,
  consider sub-sampling or a block-approximate eigen solver.
- scikit-learn trees do not natively support categorical splits;
  we handle categories via encoders and decode them with weighted voting.
"""

from __future__ import annotations

import dataclasses
import json
import math
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    RandomTreesEmbedding,
)
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_random_state

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# SciPy is optional but recommended for large-n symmetric eigen problems
try:
    from scipy.sparse.linalg import eigsh  # type: ignore
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


@dataclass
class RFAEConfig:
    # Forest
    n_estimators: int = 500
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    random_state: Optional[int] = 42
    n_jobs: Optional[int] = None

    # Mode: "supervised-classification", "supervised-regression", "unsupervised"
    mode: str = "supervised-classification"

    # Diffusion map
    latent_dim: int = 16             # d_Z
    diffusion_time: float = 1.0      # t in the paper (λ^t scaling)
    use_torch: bool = True           # if True and CUDA available, compute eigendecomp on GPU
    device: Optional[str] = None     # "cuda" | "cpu" | None (auto)

    # Decoder (k-NN)
    k_neighbors: int = 50
    distance_eps: float = 1e-8       # avoid div-by-zero in weights
    weight_power: float = 2.0        # weights ∝ 1 / (eps + dist)^p
    exact_decoder: bool = False      # if True, sample synthetic training inputs from leaf intersections
    max_trees_for_intersection: Optional[int] = 256  # to control cost of exact decoding

    # Preprocessing
    standardize_numeric: bool = True
    categorical_strategy: str = "ordinal"  # "ordinal" or "onehot" (onehot only used for downstream ML; decoding uses vote)
    handle_unknown_cats: str = "use_encoded_value"   # passed to OrdinalEncoder

    # Numerics
    eigen_solver: str = "auto"       # "auto" | "numpy" | "scipy" | "torch"
    float_dtype: str = "float64"     # "float32" or "float64"

    # Misc
    verbose: bool = True


def _infer_columns(X: Union[pd.DataFrame, np.ndarray]) -> Tuple[List[str], List[str]]:
    """Return (numeric_cols, categorical_cols)."""
    if isinstance(X, pd.DataFrame):
        cat_cols = [c for c in X.columns if pd.api.types.is_object_dtype(X[c]) or pd.api.types.is_categorical_dtype(X[c])]
        num_cols = [c for c in X.columns if c not in cat_cols]
        return num_cols, cat_cols
    else:
        # Assume all numeric if no DataFrame schema is available
        n = X.shape[1]
        return [f"x{i}" for i in range(n)], []


def _make_preprocessor(
    X: Union[pd.DataFrame, np.ndarray],
    cfg: RFAEConfig
) -> Tuple[Pipeline, List[str], List[str]]:
    num_cols, cat_cols = _infer_columns(X)

    transformers = []
    if len(num_cols) > 0:
        if cfg.standardize_numeric:
            transformers.append(("num", StandardScaler(with_mean=True, with_std=True), num_cols))
        else:
            transformers.append(("num", "passthrough", num_cols))

    if len(cat_cols) > 0:
        if cfg.categorical_strategy == "ordinal":
            oe = OrdinalEncoder(
                handle_unknown=cfg.handle_unknown_cats,
                unknown_value=-1
            )
            transformers.append(("cat", oe, cat_cols))
        else:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            transformers.append(("cat", ohe, cat_cols))

    pre = ColumnTransformer(transformers, remainder="drop")
    pipe = Pipeline([("pre", pre)])
    return pipe, num_cols, cat_cols


class RFAE(BaseEstimator, TransformerMixin):
    """
    Random-Forest Autoencoder with diffusion-map encoder and k-NN decoder.

    API
    ---
    rfae = RFAE(RFAEConfig(...))
    rfae.fit(X, y=None)       # trains the forest, builds RF kernel and spectral embedding
    Z   = rfae.encode(X)      # encodes new samples via Nyström
    Xh  = rfae.decode(Z)      # decodes latent vectors back to input space

    Notes
    -----
    - For classification/regression, pass y and set mode to the appropriate supervised mode.
    - For pure unsupervised embeddings, set mode="unsupervised" and y=None.

    - Decoding (k-NN) supports both numeric & categorical features:
        numeric   -> weighted average in input space (of training representatives)
        categorical -> weighted majority vote across neighbors
    """
    def __init__(self, config: Optional[RFAEConfig] = None):
        self.cfg = config or RFAEConfig()
        self._rng = check_random_state(self.cfg.random_state)

        # Placeholders populated after fit
        self.preprocessor_: Optional[Pipeline] = None
        self.num_cols_: List[str] = []
        self.cat_cols_: List[str] = []
        self.is_supervised_: bool = False
        self.task_: str = "classification"

        self.forest_: Any = None
        self.estimators_: List[Any] = []

        # Kernel / spectral pieces
        self.K_: Optional[np.ndarray] = None      # (n, n)
        self.evals_: Optional[np.ndarray] = None  # (d+1,) includes trivial eval (≈1.0) at index 0
        self.evecs_: Optional[np.ndarray] = None  # (n, d+1) columns aligned with evals_
        self.Z_train_: Optional[np.ndarray] = None  # (n, d) diffusion coords (w/o trivial eigenvector)
        self.Z_scaler_: float = 1.0

        # Training data (preprocessed)
        self.X_train_pre_: Optional[np.ndarray] = None   # numeric + encoded categorical
        self.X_train_df_: Optional[pd.DataFrame] = None  # original typed DataFrame if available
        self.n_samples_: int = 0
        self.n_features_: int = 0

        # Structures for fast K0 computation
        self._leaf_index_per_tree_: List[Dict[int, np.ndarray]] = []  # per tree: leaf_id -> row indices in training
        self._leaf_size_per_tree_: List[Dict[int, int]] = []          # per tree: leaf_id -> count

        # Exact decoding (optional)
        self._synthetic_train_X_: Optional[np.ndarray] = None

    # ---------------------------- Public API ---------------------------- #

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Union[np.ndarray, Iterable]] = None) -> "RFAE":
        """Fit the RF, construct the RF kernel, and compute the diffusion-map embedding on training data."""
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=[f"x{i}" for i in range(np.asarray(X).shape[1])])
        self.X_train_df_ = X_df.copy()

        # Preprocess
        self.preprocessor_, self.num_cols_, self.cat_cols_ = _make_preprocessor(X_df, self.cfg)
        Xp = self.preprocessor_.fit_transform(X_df)
        Xp = np.asarray(Xp, dtype=self.cfg.float_dtype)
        self.X_train_pre_ = Xp
        self.n_samples_, self.n_features_ = Xp.shape

        # Mode & RF
        self.is_supervised_ = self.cfg.mode.startswith("supervised")
        if self.is_supervised_:
            if y is None:
                raise ValueError("Supervised mode selected but y is None.")
            y = np.asarray(y)
            if self.cfg.mode == "supervised-classification":
                self.task_ = "classification"
                self.forest_ = RandomForestClassifier(
                    n_estimators=self.cfg.n_estimators,
                    max_depth=self.cfg.max_depth,
                    min_samples_leaf=self.cfg.min_samples_leaf,
                    n_jobs=self.cfg.n_jobs,
                    random_state=self.cfg.random_state,
                )
            elif self.cfg.mode == "supervised-regression":
                self.task_ = "regression"
                self.forest_ = RandomForestRegressor(
                    n_estimators=self.cfg.n_estimators,
                    max_depth=self.cfg.max_depth,
                    min_samples_leaf=self.cfg.min_samples_leaf,
                    n_jobs=self.cfg.n_jobs,
                    random_state=self.cfg.random_state,
                )
            else:
                raise ValueError(f"Unknown supervised mode: {self.cfg.mode}")
            self.forest_.fit(Xp, y)
            self.estimators_ = list(self.forest_.estimators_)
        else:
            self.task_ = "unsupervised"
            rte = RandomTreesEmbedding(
                n_estimators=self.cfg.n_estimators,
                max_depth=self.cfg.max_depth,
                n_jobs=self.cfg.n_jobs,
                random_state=self.cfg.random_state,
            )
            rte.fit(Xp)  # no y
            # We only need the underlying decision trees
            self.forest_ = rte
            self.estimators_ = list(rte.estimators_)

        if self.cfg.verbose:
            print(f"[RFAE] Fitted forest with {len(self.estimators_)} trees; building RF kernel...")

        # Precompute leaf -> indices (and sizes) per tree for K and K0
        self._leaf_index_per_tree_ = []
        self._leaf_size_per_tree_ = []
        for t_idx, est in enumerate(self.estimators_):
            leaves = est.apply(self.X_train_pre_)
            leaf_to_idx: Dict[int, np.ndarray] = {}
            for lid in np.unique(leaves):
                idx = np.where(leaves == lid)[0]
                leaf_to_idx[int(lid)] = idx
            self._leaf_index_per_tree_.append(leaf_to_idx)
            self._leaf_size_per_tree_.append({lid: len(idx) for lid, idx in leaf_to_idx.items()})

        # Build K (dense)
        self.K_ = self._build_rf_kernel_train()
        if self.cfg.verbose:
            row_sums = self.K_.sum(axis=1)[:5]
            print(f"[RFAE] RF kernel built. Example row sums (should be ~1.0): {np.round(row_sums, 3)}")

        # Spectral decomposition and diffusion map
        self.evals_, self.evecs_ = self._eigendecompose(self.K_, top_k=self.cfg.latent_dim + 1)
        # Drop the trivial eigenpair (λ≈1, constant vector)
        evals_nontrivial = self.evals_[1: self.cfg.latent_dim + 1]
        evecs_nontrivial = self.evecs_[:, 1: self.cfg.latent_dim + 1]
        # Diffusion coordinates: Z = sqrt(n) * V * Λ^t
        self.Z_scaler_ = float(np.sqrt(self.n_samples_))
        Z = (self.Z_scaler_ * evecs_nontrivial) * (evals_nontrivial ** self.cfg.diffusion_time)
        self.Z_train_ = np.asarray(Z, dtype=self.cfg.float_dtype)

        # Optional synthetic representatives for exact decoding
        if self.cfg.exact_decoder:
            if self.cfg.verbose:
                print("[RFAE] Precomputing synthetic training inputs from leaf intersections (exact decoder)...")
            self._synthetic_train_X_ = self._build_synthetic_training_points()

        return self

    def encode(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Nyström out-of-sample encoding into diffusion space."""
        check_is_fitted(self, ["Z_train_", "evals_", "evecs_", "K_"])
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.X_train_df_.columns)
        Xp = np.asarray(self.preprocessor_.transform(X_df), dtype=self.cfg.float_dtype)

        # Compute K0 (m x n) using the RF kernel between test and training samples
        K0 = self._build_rf_kernel_test(Xp)
        # Nyström: Z0 = sqrt(n) * K0 * V * Λ^{t-1}
        # Equivalent to: Z0 = K0 @ (Z_train / Λ), because Z_train = sqrt(n) * V * Λ^t
        evals_nontrivial = self.evals_[1: self.cfg.latent_dim + 1]
        # safeguard
        lam_inv = np.where(evals_nontrivial > 1e-12, evals_nontrivial ** (self.cfg.diffusion_time - 1.0), 0.0)
        # Compute Z0 = K0 @ (sqrt(n) * V * Λ^t * Λ^{-1}) = K0 @ (sqrt(n) * V * Λ^{t-1})
        # But we have Z_train_ already = sqrt(n) * V * Λ^t  (d columns). So Z0 = K0 @ Z_train_ @ diag(Λ^{-1})
        Z0 = K0 @ (self.Z_train_ * (evals_nontrivial ** -1.0))
        return np.asarray(Z0, dtype=self.cfg.float_dtype)

    def decode(self, Z: np.ndarray, k: Optional[int] = None) -> Union[np.ndarray, pd.DataFrame]:
        """Decode latent vectors Z back to input space using k-NN with weighted averaging / voting.
        Returns a pandas DataFrame if the input X at fit-time was a DataFrame (to preserve dtypes).
        """
        check_is_fitted(self, ["Z_train_", "X_train_pre_"])
        k = k or self.cfg.k_neighbors
        Z = np.asarray(Z, dtype=self.cfg.float_dtype)

        nbrs = NearestNeighbors(n_neighbors=min(k, len(self.Z_train_)), algorithm="auto")
        nbrs.fit(self.Z_train_)
        dists, inds = nbrs.kneighbors(Z, return_distance=True)  # shapes (m, k)

        # weights ∝ 1 / (eps + dist)^p, normalized
        w = 1.0 / np.maximum(dists, self.cfg.distance_eps) ** self.cfg.weight_power
        w = (w / w.sum(axis=1, keepdims=True)).astype(self.cfg.float_dtype)  # (m, k)

        # Representatives to average from
        if self.cfg.exact_decoder and self._synthetic_train_X_ is not None:
            Xrep = self._synthetic_train_X_
        else:
            Xrep = self.X_train_pre_  # fast approximation

        X_rep_neighbors = Xrep[inds]  # (m, k, d_x_pre)
        # Continuous part: always present at the front of the preprocessed array
        X_cont_recon = np.einsum("mk,mkd->md", w, X_rep_neighbors)

        # Recompose frame from numeric + categorical voting
        # Identify categorical blocks in the transformed space
        # For ordinal: each cat is a single column in transformed space.
        # For onehot: will be multiple columns. We'll recover from preprocessor metadata.
        decoded_pre = X_cont_recon  # start with continuous approximation

        # Build inverse to raw space
        # We decode categories separately (vote on original category labels if possible).
        if len(self.cat_cols_) > 0:
            # Get the original (training) categorical columns for neighbors in raw space
            # We'll vote on labels, not on encoded values
            X_train_raw = self.X_train_df_.reset_index(drop=True)
            cat_vals_neighbors = X_train_raw.loc[inds.flatten(), self.cat_cols_].to_numpy().reshape(Z.shape[0], -1, len(self.cat_cols_))
            # Weighted vote per categorical feature
            cats_decoded = []
            for j, col in enumerate(self.cat_cols_):
                vals_j = cat_vals_neighbors[:, :, j]  # (m, k) of labels
                winners = []
                for i in range(Z.shape[0]):
                    # weighted counts
                    weights_i = w[i]
                    labels_i = vals_j[i]
                    # aggregate weights by label value
                    label_to_w = {}
                    for lbl, ww in zip(labels_i, weights_i):
                        label_to_w[lbl] = label_to_w.get(lbl, 0.0) + float(ww)
                    # winner with tie broken randomly but deterministically via rng
                    top_w = max(label_to_w.values())
                    candidates = [lbl for lbl, ww in label_to_w.items() if abs(ww - top_w) < 1e-12]
                    winners.append(self._rng.choice(candidates))
                cats_decoded.append(np.array(winners, dtype=object))
            cats_decoded = np.column_stack(cats_decoded)
        else:
            cats_decoded = None

        # Inverse transform numeric part back to raw space
        # The ColumnTransformer inversion is not natively supported; we reconstruct via a DataFrame.
        # We'll build a DataFrame with numeric cols from decoded_pre (taking the appropriate slice).
        # NOTE: For onehot encoding we cannot directly invert without additional bookkeeping;
        #       recommend `categorical_strategy="ordinal"` if you want exact inverse in this path.
        out_df = pd.DataFrame(index=range(Z.shape[0]))
        if len(self.num_cols_) > 0:
            # Recover the numeric slice from the preprocessor
            # Since we used a ColumnTransformer, the numeric block is first (by construction in _make_preprocessor).
            n_num = len(self.num_cols_)
            out_df[self.num_cols_] = decoded_pre[:, :n_num]

        if cats_decoded is not None:
            for j, col in enumerate(self.cat_cols_):
                out_df[col] = cats_decoded[:, j]

        return out_df

    # ---------------------------- Internals ---------------------------- #

    def _build_rf_kernel_train(self) -> np.ndarray:
        """Construct K for the training set based on Eq. (1) in the paper:
            k_RF(x, x') = (1/B) sum_b k_b(x,x') / sum_i k_b(x, x_i)
        which reduces to adding 1/leaf_size for pairs of training samples that co-locate in a tree leaf.
        """
        n = self.n_samples_
        K = np.zeros((n, n), dtype=self.cfg.float_dtype)
        B = float(len(self.estimators_))

        for t_idx, leaf_to_idx in enumerate(self._leaf_index_per_tree_):
            for lid, idx in leaf_to_idx.items():
                sz = len(idx)
                if sz == 0:
                    continue
                K[np.ix_(idx, idx)] += (1.0 / sz)

        K /= B
        # Guarantee symmetry and non-negativity within numerical tolerance
        K = (K + K.T) * 0.5
        K[K < 0] = 0.0
        # Row-normalization should already hold by construction; we keep as-is since K is doubly-stochastic.
        return K

    def _build_rf_kernel_test(self, X_test_pre: np.ndarray) -> np.ndarray:
        """Compute K0 (m x n) for out-of-sample X_test against training data."""
        m, n = X_test_pre.shape[0], self.n_samples_
        K0 = np.zeros((m, n), dtype=self.cfg.float_dtype)
        B = float(len(self.estimators_))

        for est, leaf_to_idx, leaf_sizes in zip(self.estimators_, self._leaf_index_per_tree_, self._leaf_size_per_tree_):
            test_leaves = est.apply(X_test_pre)
            # For each test sample, add 1/leaf_size to positions of training samples that share the leaf
            for i, lid in enumerate(test_leaves):
                idx = leaf_to_idx.get(int(lid))
                if idx is None:
                    continue
                sz = leaf_sizes[int(lid)]
                if sz > 0:
                    K0[i, idx] += (1.0 / sz)

        K0 /= B
        # Ensure rows sum to ~1 (numerical jitter possible)
        row_sums = K0.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        K0 /= row_sums
        return K0

    def _eigendecompose(self, K: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute leading eigenpairs of symmetric PSD K (descending order). Drops no vectors here."""
        solver = self.cfg.eigen_solver
        if solver == "auto":
            # Choose based on availability & size
            if TORCH_AVAILABLE and self.cfg.use_torch:
                solver = "torch"
            elif SCIPY_AVAILABLE and K.shape[0] >= 512:
                solver = "scipy"
            else:
                solver = "numpy"

        if solver == "torch" and TORCH_AVAILABLE:
            device = self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
            tK = torch.tensor(K, dtype=torch.float64 if self.cfg.float_dtype == "float64" else torch.float32, device=device)
            # Full eigh; for very large n, consider Lanczos via torch.lobpcg
            evals, evecs = torch.linalg.eigh(tK)  # ascending
            evals = evals.detach().cpu().numpy()
            evecs = evecs.detach().cpu().numpy()
        elif solver == "scipy" and SCIPY_AVAILABLE:
            # Use eigsh for top_k largest algebraic eigenvalues
            # Shift-invert not required; symmetric PSD.
            # We ask for k=min(top_k, n-1) eigenpairs.
            k = min(top_k, K.shape[0] - 1)
            evals, evecs = eigsh(K, k=k, which="LA")
        else:
            evals, evecs = np.linalg.eigh(K)  # ascending

        # Sort in descending order
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        # If eigsh returned fewer than requested, pad (shouldn't happen in typical cases)
        if evecs.shape[1] < top_k:
            warnings.warn(f"Requested top_k={top_k} but got only {evecs.shape[1]} eigenpairs.")
        return evals, evecs

    # ------ Exact decoder helpers (optional; can be heavy for large forests) ------ #

    def _build_synthetic_training_points(self) -> np.ndarray:
        """For each training sample, create a synthetic representative by sampling uniformly from the
        intersection of leaf-defined hyperrectangles across trees (the 'maximum compatible rule').
        This can be computationally heavy; we allow sub-sampling of trees to control cost.
        """
        Xp = self.X_train_pre_
        n, d = Xp.shape
        # Initialize bounds with global min/max of training data
        global_min = np.min(Xp, axis=0)
        global_max = np.max(Xp, axis=0)

        # Use a subset of trees if requested, to cap compute cost
        if self.cfg.max_trees_for_intersection is not None:
            trees = self.estimators_[: self.cfg.max_trees_for_intersection]
        else:
            trees = self.estimators_

        # For each sample, intersect constraints from each tree along its path
        lows = np.tile(global_min, (n, 1)).astype(self.cfg.float_dtype)
        highs = np.tile(global_max, (n, 1)).astype(self.cfg.float_dtype)

        for t_idx, est in enumerate(trees):
            tree = est.tree_
            # decision_path: sparse matrix [n_samples, n_nodes]; True where node visited
            path = est.decision_path(Xp)  # csr_matrix
            features = tree.feature
            thresholds = tree.threshold

            for i in range(n):
                node_indices = path.indices[path.indptr[i]: path.indptr[i+1]]
                # Traverse nodes (excluding leaves where feature == _tree.TREE_UNDEFINED)
                for node in node_indices:
                    feat = features[node]
                    if feat < 0:
                        continue  # leaf
                    thr = thresholds[node]
                    # Determine direction taken
                    # We can use children_left/right to infer, but comparing Xp[i, feat] to thr is simpler
                    if Xp[i, feat] <= thr:
                        # went left: x[feat] <= thr
                        highs[i, feat] = min(highs[i, feat], thr)
                    else:
                        # went right: x[feat] > thr
                        lows[i, feat] = max(lows[i, feat], np.nextafter(thr, np.inf))  # open interval

        # Sample uniformly within [low, high] (clip to ensure feasibility)
        lows = np.minimum(lows, highs)
        widths = np.maximum(highs - lows, 0.0)
        U = self._rng.uniform(size=(n, d)).astype(self.cfg.float_dtype)
        X_syn = lows + U * widths
        return X_syn

    # ---------------------------- Persistence ---------------------------- #

    def save(self, path: Union[str, Path]) -> None:
        """Save model to disk (JSON + npz)."""
        check_is_fitted(self, ["Z_train_", "evecs_", "evals_"])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "cfg": dataclasses.asdict(self.cfg),
            "num_cols": self.num_cols_,
            "cat_cols": self.cat_cols_,
            "is_supervised": self.is_supervised_,
            "task": self.task_,
            "n_samples": self.n_samples_,
            "n_features": self.n_features_,
            "Z_scaler": self.Z_scaler_,
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(meta, f, indent=2)

        np.savez_compressed(
            path.with_suffix(".npz"),
            K=self.K_,
            evals=self.evals_,
            evecs=self.evecs_,
            Z_train=self.Z_train_,
            X_train_pre=self.X_train_pre_,
        )
        # Preprocessor & forest saved via joblib to preserve sklearn objects
        try:
            import joblib  # lazy import
            joblib.dump(
                {"preprocessor": self.preprocessor_, "forest": self.forest_, "estimators": self.estimators_},
                path.with_suffix(".skjoblib"),
            )
        except Exception as e:
            warnings.warn(f"Failed to save sklearn objects with joblib: {e}")

    @staticmethod
    def load(path: Union[str, Path]) -> "RFAE":
        """Load model from disk (paired with .save)."""
        path = Path(path)
        with open(path.with_suffix(".json"), "r") as f:
            meta = json.load(f)
        cfg = RFAEConfig(**meta["cfg"])
        model = RFAE(cfg)
        model.num_cols_ = meta["num_cols"]
        model.cat_cols_ = meta["cat_cols"]
        model.is_supervised_ = meta["is_supervised"]
        model.task_ = meta["task"]
        model.n_samples_ = meta["n_samples"]
        model.n_features_ = meta["n_features"]
        model.Z_scaler_ = meta["Z_scaler"]

        blob = np.load(path.with_suffix(".npz"))
        model.K_ = blob["K"]
        model.evals_ = blob["evals"]
        model.evecs_ = blob["evecs"]
        model.Z_train_ = blob["Z_train"]
        model.X_train_pre_ = blob["X_train_pre"]

        try:
            import joblib
            objs = joblib.load(path.with_suffix(".skjoblib"))
            model.preprocessor_ = objs["preprocessor"]
            model.forest_ = objs["forest"]
            model.estimators_ = objs.get("estimators") or getattr(model.forest_, "estimators_", [])
            # Rebuild leaf index maps
            model._leaf_index_per_tree_ = []
            model._leaf_size_per_tree_ = []
            leaves_all = [est.apply(model.X_train_pre_) for est in model.estimators_]
            for leaves in leaves_all:
                leaf_to_idx: Dict[int, np.ndarray] = {}
                for lid in np.unique(leaves):
                    idx = np.where(leaves == lid)[0]
                    leaf_to_idx[int(lid)] = idx
                model._leaf_index_per_tree_.append(leaf_to_idx)
                model._leaf_size_per_tree_.append({lid: len(idx) for lid, idx in leaf_to_idx.items()})
        except Exception as e:
            warnings.warn(f"Failed to load sklearn objects with joblib: {e}")

        return model


# ---------------------------- Tiny demo ---------------------------- #

def _demo_iris():
    """Quick smoke test on Iris dataset (runs without internet)."""
    from sklearn.datasets import load_iris
    data = load_iris(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]

    cfg = RFAEConfig(
        n_estimators=300,
        max_depth=None,
        mode="supervised-classification",
        latent_dim=4,
        diffusion_time=1.0,
        k_neighbors=15,
        use_torch=False,  # small, CPU is fine
        verbose=True,
    )
    rfae = RFAE(cfg).fit(X, y)
    Z = rfae.encode(X.iloc[:5])
    Xh = rfae.decode(Z)
    print("Encoded Z shape:", Z.shape)
    print("Decoded Xh (head):")
    print(Xh.head())


if __name__ == "__main__":
    _demo_iris()

# ---------------------------- Evaluation helper ---------------------------- #

def reconstruction_distortion(
    X_true: Union[pd.DataFrame, np.ndarray],
    X_recon: Union[pd.DataFrame, np.ndarray],
) -> Dict[str, Any]:
    """
    Compute the distortion metric used in the paper (Sec. 5):
      - For continuous variables: 1 - R^2 (proportion of variance unexplained)
      - For categorical variables: classification error
    Returns a dict with per-type and overall means.
    """
    if isinstance(X_true, np.ndarray):
        X_true_df = pd.DataFrame(X_true, columns=[f"x{i}" for i in range(X_true.shape[1])])
    else:
        X_true_df = X_true.copy()

    if isinstance(X_recon, np.ndarray):
        X_recon_df = pd.DataFrame(X_recon, columns=X_true_df.columns)
    else:
        X_recon_df = X_recon.copy()

    assert list(X_true_df.columns) == list(X_recon_df.columns), "Column mismatch"

    num_cols = [c for c in X_true_df.columns if pd.api.types.is_numeric_dtype(X_true_df[c])]
    cat_cols = [c for c in X_true_df.columns if c not in num_cols]

    # Continuous: 1 - R^2
    cont_scores = []
    for c in num_cols:
        x = X_true_df[c].to_numpy(dtype=float)
        y = X_recon_df[c].to_numpy(dtype=float)
        var = np.var(x)
        if var < 1e-12:
            continue
        ss_res = np.mean((x - y) ** 2)
        # R^2 = 1 - MSE/Var; we want 1 - R^2 = MSE/Var
        cont_scores.append(ss_res / var)
    cont_mean = float(np.mean(cont_scores)) if len(cont_scores) > 0 else np.nan

    # Categorical: classification error
    cat_scores = []
    for c in cat_cols:
        x = X_true_df[c].astype(str).to_numpy()
        y = X_recon_df[c].astype(str).to_numpy()
        err = float(np.mean(x != y))
        cat_scores.append(err)
    cat_mean = float(np.mean(cat_scores)) if len(cat_scores) > 0 else np.nan

    # Overall: average of (cont_mean, cat_mean) ignoring NaNs
    parts = [v for v in [cont_mean, cat_mean] if not np.isnan(v)]
    overall = float(np.mean(parts)) if len(parts) > 0 else np.nan

    return {"continuous": cont_mean, "categorical": cat_mean, "overall": overall}

# ---------------------------- Optional: neural encoder (PyTorch) ---------------------------- #

class _MLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: List[int]):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [torch.nn.Linear(last, h), torch.nn.ReLU()]
            last = h
        layers += [torch.nn.Linear(last, out_dim)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _to_tensor(x: np.ndarray, device: str, dtype: torch.dtype):
    return torch.tensor(x, device=device, dtype=dtype)


def _from_tensor(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _early_stop(loss_hist: List[float], patience: int) -> bool:
    if len(loss_hist) < patience + 1:
        return False
    best = min(loss_hist[:-patience])
    return all(l >= best for l in loss_hist[-patience:])


def _progress(msg: str, enabled: bool):
    if enabled:
        print(msg)


# Attach methods to RFAE via monkey-patching to keep the top of file tidy
def rfae_fit_neural_encoder(self: RFAE,
                            hidden: List[int] = [256, 128],
                            epochs: int = 50,
                            batch_size: int = 256,
                            lr: float = 1e-3,
                            weight_decay: float = 1e-6,
                            patience: int = 10,
                            verbose: bool = True) -> "RFAE":
    """
    Train a small MLP g_theta: X_pre -> Z to approximate Nyström encoding.
    Useful when you will encode many out-of-sample points repeatedly.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Install torch to use fit_neural_encoder.")

    check_is_fitted(self, ["Z_train_", "X_train_pre_"])

    device = self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if self.cfg.float_dtype == "float32" else torch.float64

    model = _MLP(self.n_features_, self.cfg.latent_dim, hidden).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()

    X = _to_tensor(self.X_train_pre_, device, dtype)
    Z = _to_tensor(self.Z_train_, device, dtype)

    n = X.shape[0]
    idx = np.arange(n)
    loss_hist: List[float] = []

    for ep in range(1, epochs + 1):
        self._rng.shuffle(idx)
        model.train()
        ep_loss = 0.0
        for s in range(0, n, batch_size):
            b = idx[s:s+batch_size]
            xb = X[b]
            zb = Z[b]
            pred = model(xb)
            loss = loss_fn(pred, zb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach().cpu().item()) * len(b)
        ep_loss /= float(n)
        loss_hist.append(ep_loss)
        _progress(f"[RFAE][neural-encoder] epoch {ep:03d}/{epochs}  loss={ep_loss:.6f}", verbose)
        if _early_stop(loss_hist, patience):
            _progress("[RFAE][neural-encoder] early stopping", verbose)
            break

    self._nn_encoder = model
    self._nn_device = device
    self._nn_dtype = dtype
    return self


def rfae_encode_nn(self: RFAE, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """Fast encoding via the trained neural encoder (requires fit_neural_encoder)."""
    if not hasattr(self, "_nn_encoder"):
        raise RuntimeError("Neural encoder not trained. Call fit_neural_encoder first.")
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.X_train_df_.columns)
    Xp = np.asarray(self.preprocessor_.transform(X_df), dtype=self.cfg.float_dtype)
    self._nn_encoder.eval()
    with torch.no_grad():
        z = self._nn_encoder(_to_tensor(Xp, self._nn_device, self._nn_dtype))
    return _from_tensor(z)


# Bind methods
setattr(RFAE, "fit_neural_encoder", rfae_fit_neural_encoder)
setattr(RFAE, "encode_nn", rfae_encode_nn)
