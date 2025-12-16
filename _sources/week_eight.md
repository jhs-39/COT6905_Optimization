---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  name: python3
  display_name: 'Python 3'
---

# Week 8: SVD, Linear Transform, Compression

## Lecture 29: Singular value decomposition

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/TX_vooSnhm8"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

The **singular value decomposition** (SVD) is the ultimate factorization: it works for **any matrix** (square or rectangular, full rank or not) and reveals its **rank**, **fundamental subspaces**, and **condition number**.

---

### 1. SVD Form

For any $m \times n$ matrix $A$:  
\[
\boxed{A = U \Sigma V^T}
\]
- $U$: $m \times m$ orthogonal ($U^T U = I$)  
- $\Sigma$: $m \times n$ diagonal (singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$)  
- $V$: $n \times n$ orthogonal ($V^T V = I$)

The **rank** $r = \#$ of nonzero $\sigma_i$.

> **Note**: If $A$ is symmetric positive definite, SVD = eigenvalue decomposition: $A = Q \Lambda Q^T$.

---

### 2. Geometric Interpretation

- Columns of $V$ = **orthonormal basis** for row space $C(A^T)$ (input space).  
- Columns of $U$ = **orthonormal basis** for column space $C(A)$ (output space).  
- $A$ maps $v_i$ to $\sigma_i u_i$:  
  \[
  \boxed{A v_i = \sigma_i u_i}
  \]
- For $i > r$, $\sigma_i = 0$ → $v_i$ in null space $N(A)$.

Full matrix equation:  
\[
A V = U \Sigma \quad \Rightarrow \quad A = U \Sigma V^T
\]

---

### 3. Computing SVD: Via Eigenvalues of $A^T A$ and $A A^T$

\[
A^T A = (U \Sigma V^T)^T (U \Sigma V^T) = V \Sigma^T U^T U \Sigma V^T = V (\Sigma^T \Sigma) V^T
\]

- $\Sigma^T \Sigma$ = diagonal with $\sigma_i^2$ → eigenvalues of $A^T A$.  
- $V$ = eigenvectors of $A^T A$.

Similarly:  
\[
A A^T = U (\Sigma \Sigma^T) U^T
\]
- $U$ = eigenvectors of $A A^T$.

> **Key**: Eigenvalues of $A^T A$ and $A A^T$ are the same (nonzero) $\sigma_i^2$.

---

### 4. Example: Full Rank Matrix

\[
A = \begin{pmatrix} 4 & 4 \\ -3 & 3 \end{pmatrix}
\]

First, $A^T A = \begin{pmatrix} 25 & 7 \\ 7 & 25 \end{pmatrix}$

Eigenvalues: $\lambda_1 = 32$, $\lambda_2 = 18$ → $\sigma_1 = \sqrt{32} = 4\sqrt{2}$, $\sigma_2 = \sqrt{18} = 3\sqrt{2}$

Eigenvectors (normalized):  
$v_1 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix}$, $v_2 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ -1 \end{pmatrix}$

Now, $A A^T = \begin{pmatrix} 32 & 0 \\ 0 & 18 \end{pmatrix}$

Eigenvectors (normalized):  
$u_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$, $u_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$

**Note**: Signs may vary; adjust to match $A v_i = \sigma_i u_i$.

SVD:  
\[
A = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} 4\sqrt{2} & 0 \\ 0 & 3\sqrt{2} \end{pmatrix} \begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}} \end{pmatrix}^T
\]

---

### 5. Example: Rank-Deficient Matrix

\[
A = \begin{pmatrix} 4 & 3 \\ 8 & 6 \end{pmatrix} \quad (\text{rank 1})
\]

$A^T A = \begin{pmatrix} 80 & 60 \\ 60 & 45 \end{pmatrix}$

Eigenvalues: $\lambda_1 = 125$, $\lambda_2 = 0$ → $\sigma_1 = \sqrt{125} = 5\sqrt{5}$, $\sigma_2 = 0$

$v_1 = \begin{pmatrix} 0.8 \\ 0.6 \end{pmatrix}$ (normalized)

$A A^T = \begin{pmatrix} 25 & 50 \\ 50 & 100 \end{pmatrix}$

Eigenvalues: same nonzero, plus zero.

$u_1 = \frac{1}{\sqrt{5}} \begin{pmatrix} 1 \\ 2 \end{pmatrix}$ (adjust sign if needed)

SVD:  
\[
A = \frac{1}{\sqrt{5}} \begin{pmatrix} 1 & -2 \\ 2 & 1 \end{pmatrix} \begin{pmatrix} 5\sqrt{5} & 0 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} 0.8 & 0.6 \\ 0.6 & -0.8 \end{pmatrix}^T
\]

> **Note**: In lecture, signs may differ; verify $A = U \Sigma V^T$.


### Interactive SVD Explorer
Run the code below to generate a random low-rank matrix and compute its SVD. Predict the singular values and rank, then check!

```{code-cell} python
import numpy as np
from IPython.display import Markdown, display

def generate_low_rank_matrix(m=3, n=4, rank=2):
    """
    Generate a random m x n matrix with specified rank.
    """
    U = np.random.randn(m, rank)
    V = np.random.randn(n, rank)
    A = U @ V.T
    return A

def matrix_to_latex(M):
    rows = [r" & ".join([f"{x:.2f}" if abs(x) > 1e-10 else "0" for x in row]) for row in M]
    return r"\begin{pmatrix} " + r" \\ ".join(rows) + r"\end{pmatrix}"

# Generate matrix
A = generate_low_rank_matrix()

# Compute SVD
U_svd, S, Vt = np.linalg.svd(A, full_matrices=True)
Sigma = np.zeros((U_svd.shape[1], Vt.shape[0]))
np.fill_diagonal(Sigma, S)

# Display matrix
markdown = f"**Matrix A** (rank ~2):\n$$\n{matrix_to_latex(A)}\n$$\n\n"
markdown += "Predict:\n- Singular values (σ).\n- Rank.\n- Bases for U and V.\n\n"
display(Markdown(markdown))

# Reveal answers (students uncomment after predicting)
# markdown = f"**SVD**:\n"
# markdown += f"- Singular values: {np.round(S, 3).tolist()}\n"
# markdown += f"- Rank: {np.sum(S > 1e-10)}\n"
# markdown += f"- U:\n$$\n{matrix_to_latex(np.round(U_svd, 3))}\n$$\n"
# markdown += f"- Σ:\n$$\n{matrix_to_latex(np.round(Sigma, 3))}\n$$\n"
# markdown += f"- V^T:\n$$\n{matrix_to_latex(np.round(Vt, 3))}\n$$\n\n"
# display(Markdown(markdown))
```

## Lecture 30: Linear transformations and their matrices

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/Ts3o2I8_Mxc"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

## Lecture 30: Linear transformations and their matrices

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/Ts3o2I8_Mxc"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

A **linear transformation** $T: \mathbb{R}^n \to \mathbb{R}^m$ is a function between vector spaces that preserves vector addition and scalar multiplication.

---

### 1. Definition

$T$ is **linear** if for all vectors $v, w$ and scalars $c$:

\[
\boxed{T(v + w) = T(v) + T(w)}, \quad \boxed{T(c v) = c T(v)}
\]

Equivalently:

\[
T(c v + d w) = c T(v) + d T(w)
\]

Consequences:

- $T(0) = 0$

---

### 2. Examples and Non-Examples

| Transformation | Linear? | Reason |
|---------------|---------|--------|
| Projection onto a line in $\mathbb{R}^2$ | Yes | Preserves addition & scaling |
| Rotation by any angle | Yes | Distances and directions preserved in a linear way |
| Translation: $T(v) = v + v_0$ ($v_0 \neq 0$) | No | $T(0) = v_0 \neq 0$ |
| Length: $T(v) = \|v\|$ | No | $T(-2v) = 2\|v\| \neq -2 T(v)$ |

---

### 3. Matrix Representation

Every linear transformation $T: \mathbb{R}^n \to \mathbb{R}^m$ can be represented by a matrix $A$ ($m \times n$) such that

\[
\boxed{T(v) = A v}
\]

**Key idea**: To define $T$ completely, it suffices to know $T(e_1), \dots, T(e_n)$ where $\{e_1, \dots, e_n\}$ is the **standard basis** of $\mathbb{R}^n$.

The columns of $A$ are exactly these images:

\[
A = \begin{bmatrix} T(e_1) & T(e_2) & \cdots & T(e_n) \end{bmatrix}
\]

---

### 4. General Bases

Suppose we choose any basis $\{v_1, \dots, v_n\}$ for the input space and $\{w_1, \dots, w_m\}$ for the output space.

Any vector $v = c_1 v_1 + \cdots + c_n v_n$, so

\[
T(v) = c_1 T(v_1) + \cdots + c_n T(v_n)
\]

Express each $T(v_j)$ in the output basis:

\[
T(v_j) = a_{1j} w_1 + \cdots + a_{mj} w_m
\]

Then the matrix $A$ (with respect to these bases) has columns given by these coefficients.

---

### 5. Example: Projection onto a Line

Project onto the line in direction $b = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$ in $\mathbb{R}^2$.

Choose input basis:

- $v_1 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix}$ (unit vector along line)
- $v_2 = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ -1 \end{pmatrix}$ (perpendicular)

Same for output basis (standard orthonormal works too, but let's use same).

Projection leaves component along $v_1$ unchanged, drops component along $v_2$:

\[
T(v_1) = v_1, \quad T(v_2) = 0
\]

Matrix (in this basis):

\[
A = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}
\]

---

### 6. Example: From $\mathbb{R}^3$ to $\mathbb{R}^2$

Let $T\begin{pmatrix} x \\ y \\ z \end{pmatrix} = \begin{pmatrix} x + y \\ y + z \end{pmatrix}$.

Matrix (standard bases):

\[
A = \begin{pmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \end{pmatrix}
\]

Columns: $T(e_1) = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$, $T(e_2) = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$, $T(e_3) = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$.

---

### 7. Example: Differentiation (Infinite Dimensions → Finite)

Consider polynomials of degree $\leq 2$: basis $\{1, x, x^2\}$.

$T(p) = p'(x)$.

\[
T(1) = 0, \quad T(x) = 1, \quad T(x^2) = 2x
\]

Output basis $\{1, x\}$.

Coefficients:

- $T(1) = 0 \cdot 1 + 0 \cdot x$
- $T(x) = 1 \cdot 1 + 0 \cdot x$
- $T(x^2) = 0 \cdot 1 + 2 \cdot x$

Matrix (2×3):

\[
A = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 2 \end{pmatrix}
\]

---

### Code: Visualizing a Linear Transformation

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

# Define a 2x2 matrix (linear transformation)
A = np.array([[2, 1],
              [0, 1.5]])  # Shear + scale

# Grid of points in unit square
x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
X, Y = np.meshgrid(x, y)
points = np.stack([X.ravel(), Y.ravel()])

# Apply transformation
transformed = A @ points
X_t = transformed[0].reshape(10, 10)
Y_t = transformed[1].reshape(10, 10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Original
ax1.plot([0,1,1,0,0], [0,0,1,1,0], 'b', linewidth=2)
ax1.grid(True)
ax1.set_xlim(-0.5, 1.5); ax1.set_ylim(-0.5, 1.5)
ax1.set_aspect('equal')
ax1.set_title('Original Unit Square')

# Transformed
ax2.plot([0, A[:,0][0], A[:,0][0]+A[:,1][0], A[:,1][0], 0],
         [0, A[:,0][1], A[:,0][1]+A[:,1][1], A[:,1][1], 0], 'r', linewidth=2)
ax2.grid(True)
ax2.set_xlim(-0.5, 3.5); ax2.set_ylim(-0.5, 3)
ax2.set_aspect('equal')
ax2.set_title('After Transformation T(v) = A v')

plt.show()
```

## Lecture 31: Change of basis; image compression

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/0h43aV4aH7I"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

## Lecture 31: Change of basis; image compression

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/0h43aV4aH7I"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

We explore how **changing the basis** can reveal structure in data and enable **compression** — representing the same information with fewer numbers.

---

### 1. Compression: Lossless vs Lossy

- **Lossless**: Exact reconstruction (e.g., ZIP, PNG).
- **Lossy**: Approximate reconstruction, but much higher compression (e.g., JPEG, MP3).

A 512×512 grayscale image (8-bit pixels):  
$n = 512^2 = 262\,144$ numbers, each 0–255 → ~2 million bits.

In the **standard pixel basis**, each basis vector has a 1 in one pixel and 0 elsewhere — highly inefficient for correlated images (e.g., a blackboard with smooth regions).

---

### 2. Change of Basis for Compression

Idea: Represent the image $x \in \mathbb{R}^n$ in a new basis $\{v_1, \dots, v_n\}$ where most coefficients are small or zero.

Let $W$ be the matrix with columns $v_1, \dots, v_n$ (orthonormal → $W^T W = I$).

Then:

\[
\boxed{x = W c}, \quad c = W^T x
\]

- $c$: coefficients in new basis (lossless transform).
- Compress by keeping only largest $|c_i|$ → $\hat{c}$ (many zeros).
- Reconstruct: $\hat{x} = W \hat{c}$.

Good basis → few large coefficients → high compression with low error.

---

### 3. JPEG: Discrete Cosine Transform (DCT)

JPEG breaks image into 8×8 blocks (lossless tiling).

Each block uses a **cosine basis** (Fourier-like, low to high frequencies).

Low-frequency coefficients (smooth parts) are large; high-frequency (details) often small → quantized/thrown away (lossy).

---

### 4. Wavelets: An Alternative Basis

Wavelets capture both location and frequency.

Example basis for $\mathbb{R}^8$ (Haar wavelets, scaled):

- Constant: $[1,1,1,1,1,1,1,1]$
- Coarse difference: $[1,1,1,1,-1,-1,-1,-1]$
- Finer differences, etc.

Advantages: Better for images with sharp edges.

---

### 5. Formal Change of Basis

Suppose matrix $A$ represents a linear transformation in the **standard basis**.

We change to new bases:

- Input basis columns in matrix $M$ (standard → new input basis)
- Output basis columns in matrix $N$ (standard → new output basis)

Then the matrix in the new bases is:

\[
\boxed{B = N^{-1} A M}
\]

If bases are the same ($M = N$), then $A$ and $B$ are **similar**:

\[
\boxed{B = M^{-1} A M}
\]

Same eigenvalues, different eigenvectors.

---

### 6. Compression Pipeline

\[
\text{Image } x 
\xrightarrow{\text{change basis}} c = W^T x 
\xrightarrow{\text{quantize / threshold}} \hat{c} 
\xrightarrow{\text{inverse}} \hat{x} = W \hat{c}
\]

For orthonormal $W$: inverse is just $W^T$ → fast!

---

### Code: Simple 1D Compression Demo (DCT-like)

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct, idct

# Synthetic smooth signal (highly compressible in DCT basis)
n = 256
x = np.sin(np.linspace(0, 4*np.pi, n)) + 0.5*np.cos(np.linspace(0, 20*np.pi, n))

# DCT (orthonormal basis change)
c = dct(x, norm='ortho')

# Keep only top k coefficients
k_values = [5, 20, 50, 256]
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(x, 'k', label='Original')
plt.title('Original Signal')
plt.legend()

plt.subplot(3, 1, 2)
plt.stem(c, use_line_collection=True)
plt.title('DCT Coefficients')
plt.xlim(0, 50)

plt.subplot(3, 1, 3)
for k in k_values:
    c_hat = np.copy(c)
    c_hat[k:] = 0
    x_hat = idct(c_hat, norm='ortho')
    compression_ratio = n / k
    plt.plot(x_hat, label=f'k={k} (compression ~{compression_ratio:.0f}x)')

plt.title('Reconstructions')
plt.legend()
plt.tight_layout()
plt.show()
```

## Lecture 32: Left and right inverses; pseudoinverse

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/Go2aLo7ZOlU"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

Not every matrix has a two-sided inverse, but **every** matrix has a **pseudoinverse** — the closest thing to an inverse, built from the SVD.

---

### 1. Two-Sided Inverse

$A$ ($n \times n$) has an inverse $A^{-1}$ if

\[
\boxed{A A^{-1} = A^{-1} A = I}
\]

Exists $\iff$ full rank ($r = n$), null space $\{0\}$.

---

### 2. Left Inverse (Tall Matrix, Full Column Rank)

$A$: $m \times n$ with $m > n$ and $\rank(A) = n$ (full column rank).

- $N(A) = \{0\}$ → at most one solution to $Ax = b$.
- $A^T A$ ($n \times n$) is symmetric positive definite → invertible.

**Left inverse**:

\[
\boxed{A_L^{-1} = (A^T A)^{-1} A^T}
\]

Check:

\[
A_L^{-1} A = (A^T A)^{-1} A^T A = I_n
\]

Cannot have $A A_L^{-1} = I_m$ (wrong size).

---

### 3. Right Inverse (Wide Matrix, Full Row Rank)

$A$: $m \times n$ with $m < n$ and $\rank(A) = m$ (full row rank).

- $N(A^T) = \{0\}$ → $b$ must be in $C(A)$ for consistency; then infinitely many solutions.
- $A A^T$ ($m \times m$) invertible.

**Right inverse**:

\[
\boxed{A_R^{-1} = A^T (A A^T)^{-1}}
\]

Check:

\[
A A_R^{-1} = A A^T (A A^T)^{-1} = I_m
\]

---

### 4. Projections Revisited

- Projection onto column space $C(A)$:

\[
\boxed{P = A (A^T A)^{-1} A^T}
\]

  (If full column rank, this is the left inverse.)

- Projection onto row space $C(A^T)$:

\[
\boxed{P = A^T (A A^T)^{-1} A}
\]

  (If full row rank, this is the right inverse.)

---

### 5. Pseudoinverse (For Any Matrix)

The **Moore-Penrose pseudoinverse** $A^+$ satisfies:

- $A A^+ A = A$
- $A^+ A A^+ = A^+$
- $(A A^+)^T = A A^+$, $(A^+ A)^T = A^+ A$

**Construction via SVD**:

\[
A = U \Sigma V^T, \quad \rank(A) = r
\]

$\Sigma$: $m \times n$, nonzero singular values $\sigma_1, \dots, \sigma_r$ on diagonal.

Pseudoinverse of $\Sigma$:

\[
\Sigma^+ : n \times m, \quad (\Sigma^+)_{ii} = 
\begin{cases}
1/\sigma_i & i \leq r \\
0 & i > r
\end{cases}
\]

Then:

\[
\boxed{A^+ = V \Sigma^+ U^T}
\]

Properties:

- If $A$ invertible → $A^+ = A^{-1}$
- If full column rank → $A^+ = (A^T A)^{-1} A^T$ (left inverse)
- If full row rank → $A^+ = A^T (A A^T)^{-1}$ (right inverse)

Geometrically: $A^+$ maps from column space back to row space, inverting the nonzero singular directions.

---

### 6. Least-Squares and Minimum-Norm Solutions

- Overdetermined ($m > n$): $A^+ b = \hat{x}$ → least-squares solution ($\min \|Ax - b\|$)
- Underdetermined ($m < n$): $A^+ b = \hat{x}$ → minimum-norm solution among all solutions

---

### Code: Compute Pseudoinverse and Solve Systems

```{code-cell} python
import numpy as np
from IPython.display import Markdown, display

# Example: tall matrix (full column rank)
A_tall = np.array([[1, 0],
                   [1, 1],
                   [1, 2]], dtype=float)
b_tall = np.array([0, 1, 3])

# Example: wide matrix (full row rank)
A_wide = np.array([[1, 2, 3],
                   [4, 5, 6]], dtype=float)
b_wide = np.array([1, 2])

# Pseudoinverse via np.linalg.pinv (uses SVD)
A_plus_tall = np.linalg.pinv(A_tall)
A_plus_wide = np.linalg.pinv(A_wide)

x_tall = A_plus_tall @ b_tall
x_wide = A_plus_wide @ b_wide

display(Markdown(f"""
**Tall matrix** (3×2, full column rank):

$$ A = \\begin{{pmatrix}} 1 & 0 \\\\ 1 & 1 \\\\ 1 & 2 \\end{{pmatrix}} $$, 
$$ b = \\begin{{pmatrix}} 0 \\\\ 1 \\\\ 3 \\end{{pmatrix}} $$

Least-squares solution: $$ \\hat{{x}} = {np.round(x_tall, 3).tolist()} $$

Residual $$ \\|A\\hat{{x}} - b\\| \\approx {np.linalg.norm(A_tall @ x_tall - b_tall):.3f} $$

**Wide matrix** (2×3, full row rank):

$$ A = \\begin{{pmatrix}} 1 & 2 & 3 \\\\ 4 & 5 & 6 \\end{{pmatrix}} $$, 
$$ b = \\begin{{pmatrix}} 1 \\\\ 2 \\end{{pmatrix}} $$

Minimum-norm solution: $$ \\hat{{x}} = {np.round(x_wide, 3).tolist()} $$

Norm $$ \\|\\hat{{x}}\\| \\approx {np.linalg.norm(x_wide):.3f} $$
"""))
```

## Lecture 33: Recap

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/RWvi4Vx4CDc"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

No lecture notes or exercises for recap, but feel free to watch and review!