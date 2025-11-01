# Random-Forest Autoencoder (RFAE) — PyTorch Implementation


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/AMIRMOHAMMAD-OSS/Randomforest-autoencoders-pytorch-/blob/main/RFAE_Colab.ipynb)


A clean, user-friendly Python implementation of **Autoencoding Random Forests (RFAE)**: a diffusion‑map encoder built from a Random‑Forest kernel with out‑of‑sample **Nyström** extension, and a **k‑NN decoder** (weighted averaging for continuous features, weighted majority vote for categoricals).

> This repository provides a single‑file module: **`rfae.py`** (drop‑in ready).

---

## 📚 References

- **Autoencoding Random Forests.** *Binh Duc Vu, Jan Kapar, Marvin N. Wright, David S. Watson.* arXiv:2505.21441 (2025).  
  arXiv: https://arxiv.org/abs/2505.21441 (PDF: https://arxiv.org/pdf/2505.21441)

- **Reference code (R):** Official RFAE research repo by the authors:  
  https://github.com/bips-hb/RFAE

- Prior related tree‑autoencoder work: **eForest** (*AutoEncoder by Forest*, AAAI 2018):  
  https://ojs.aaai.org/index.php/AAAI/article/view/11732

> This Python module re‑implements the RFAE algorithm for the PyTorch + scikit‑learn ecosystem while keeping the paper’s design and notation.

---

## 🧠 Method Overview

RFAE treats a random forest as a data‑adaptive geometry:

1. **RF Kernel (training):**  
   Build a doubly‑stochastic kernel \(K \in \mathbb{R}^{n \times n}\) by averaging per‑tree co‑location in leaves and normalizing by leaf size:
   \[
     K_{ij} \;=\; \frac{1}{B} \sum_{b=1}^B \frac{\mathbf{1}\{i,j\text{ share a leaf in tree }b\}}{\#\{ \text{train points in that leaf} \}}
   \]
   Intuition: points that frequently land in the same small leaves across trees have high affinity.

2. **Diffusion‑Map Encoder:**  
   Compute leading eigenpairs \(K V = V \Lambda\). Drop the trivial constant eigenvector, and define **diffusion coordinates**
   \[
   Z_{\text{train}} \;=\; \sqrt{n}\, V \, \Lambda^{t},
   \]
   with diffusion time \(t \ge 0\).

3. **Nyström for out‑of‑sample:**  
   For new points \(X_0\), compute the cross‑kernel \(K_0\) against the training set using the same forest. Map with
   \[
     Z_0 \;=\; K_0\, Z_{\text{train}} \, \Lambda^{-1}
     \quad\big(\text{equivalently } \sqrt{n}K_0 V \Lambda^{t-1}\big).
   \]

4. **Decoder (recommended / fast):**  
   **k‑NN in latent space**: inverse‑distance weights; average for continuous features and vote for categoricals. This is the paper’s preferred fast/accurate path with universal consistency guarantees.

5. **Decoder (optional / exact):**  
   Build **synthetic representatives** by sampling uniformly from the intersection of per‑tree leaf boxes along a sample’s decision path (“maximum compatible rule”), then decode from those representatives. This is more faithful but costlier.

---

##  Features

- Supervised **classification/regression** or **unsupervised** (completely random forest)
- Mixed‑type tabular data (numeric + categorical), with automatic dtype handling
- CPU/GPU eigendecomposition (NumPy / SciPy / PyTorch)
- Nyström out‑of‑sample encoding
- k‑NN decoder with weighted averaging + voting
- Optional “exact” decoder via leaf‑box intersections
- Optional tiny **neural encoder** (Torch MLP) to approximate Nyström for high‑throughput inference
- Save/Load (NumPy + joblib), evaluation helpers

---

##  Installation

```bash
pip install numpy pandas scikit-learn torch scipy
# Then copy rfae.py into your project, or keep it alongside your notebook/script.
```

---

##  Quickstart

```python
from rfae import RFAE, RFAEConfig, reconstruction_distortion
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris(as_frame=True)
X = data.frame.drop(columns=["target"])
y = data.frame["target"]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)

cfg = RFAEConfig(
    mode="supervised-classification",   # also: "supervised-regression", "unsupervised"
    n_estimators=500,
    latent_dim=8,
    diffusion_time=1.0,
    k_neighbors=50,
    use_torch=True,
)

model = RFAE(cfg).fit(Xtr, ytr)
Ztr = model.encode(Xtr)           # diffusion coords (train)
Zte = model.encode(Xte)           # Nyström (test)

Xhat = model.decode(Zte)          # fast k-NN decoder
print(reconstruction_distortion(Xte, Xhat))
```

**Unsupervised mode** (no labels):
```python
cfg = RFAEConfig(mode="unsupervised", n_estimators=800, latent_dim=16)
model = RFAE(cfg).fit(X)      # y not needed
Z = model.encode(X_new)
Xhat = model.decode(Z)
```

**Neural encoder** (optional acceleration):
```python
# After model.fit(...)
model.fit_neural_encoder(hidden=[256, 128], epochs=50, lr=1e-3, patience=8)
Z_fast = model.encode_nn(X_new)  # faster approximation to Nyström
```

**Persist & reload:**
```python
model.save("my_rfae")
loaded = RFAE.load("my_rfae")
```

---

##  API Summary

### `RFAEConfig`
Key knobs:
- **Forest:** `n_estimators`, `max_depth`, `min_samples_leaf`, `mode`, `n_jobs`
- **Diffusion map:** `latent_dim`, `diffusion_time`, `eigen_solver`, `use_torch`, `float_dtype`
- **Decoder:** `k_neighbors`, `weight_power`, `exact_decoder`, `max_trees_for_intersection`
- **Preprocessing:** `standardize_numeric`, `categorical_strategy` (`"ordinal"`/`"onehot"`)

### `RFAE.fit(X, y=None)`
Trains forest, builds \(K\), runs eigendecomposition, caches \(Z_{\text{train}}\).

### `RFAE.encode(X)`
Nyström out‑of‑sample mapping.

### `RFAE.decode(Z)`
k‑NN decoder (weighted average for numeric, weighted vote for categoricals).

### `RFAE.fit_neural_encoder(...)` / `RFAE.encode_nn(...)`
Small MLP to approximate Nyström when you have many encodes.

### `save(path)` / `load(path)`
Persist and restore (JSON/NPZ + joblib for the sklearn pipeline/forest).

### `reconstruction_distortion(X_true, X_recon)`
Paper’s tabular distortion metric: mean of \(1-R^2\) for continuous & error rate for categoricals.

---

##  Implementation Notes

- **Kernel build:** per‑tree leaf membership is pre‑indexed for fast train/train and test/train kernel assembly.
- **Stability:** ensures symmetry/non‑negativity; uses `eigsh` for large \(n\) when SciPy is present.
- **Categoricals:** recommended `categorical_strategy="ordinal"` for invertibility; decoder always votes on **original labels**.
- **Scalability:** dense \(n\times n\) kernel ⇒ memory \(O(n^2)\). For \(n\gg 10^4\), sub‑sample or use block‑approx. eigensolvers.

---

## 📎 Citation

If you use this implementation in academic work, please cite the original paper:

```
@article{vu2025autoencodingrf,
  title   = {Autoencoding Random Forests},
  author  = {Vu, Binh Duc and Kapar, Jan and Wright, Marvin N. and Watson, David S.},
  journal = {arXiv preprint arXiv:2505.21441},
  year    = {2025}
}
```

And consider referencing this PyTorch implementation as:
```
@software{rfae_torch_2025,
  title  = {Random-Forest Autoencoder (RFAE) -- PyTorch implementation},
  author = {<your name or organization here>},
  year   = {2025},
  url    = {https://github.com/bips-hb/RFAE}  # or your fork URL
}
```

---

##  Minimal Demo

Run the built-in smoke test:

```bash
python rfae.py
```

It trains on Iris, prints embedding shapes, and shows a quick reconstruction.

---

##  License

MIT. See header in `rfae.py`.

---

##  Acknowledgments

- The RFAE concept, analysis, and experimental protocol are due to **Binh Duc Vu, Jan Kapar, Marvin N. Wright, and David S. Watson**. This repo simply offers a Pythonic re‑implementation for wider adoption.
