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

# Week 7: Symmetric and Complex Matrices

## Lecture 25: Symmetric matrices and positive definiteness

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/UCc9q_cAhho"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

Symmetric matrices are the most important class in applications: they have **real eigenvalues**, **orthogonal eigenvectors**, and lead to the beautiful **spectral theorem**.

---

### 1. Symmetric Matrix

\[
\boxed{A = A^T}
\]

---

### 2. Key Properties

For real symmetric $A$:

- All eigenvalues $\lambda_i$ are **real**.
- Eigenvectors belonging to different eigenvalues are **orthogonal**.
- We can choose a full set of **orthonormal eigenvectors** $\{q_1, \dots, q_n\}$.

Let $Q = [q_1 \cdots q_n]$ (orthogonal matrix, $Q^T Q = I$).

Then:

\[
\boxed{A = Q \Lambda Q^T}, \quad \Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_n)
\]

This is the **spectral theorem** (or **principal axis theorem**).

---

### 3. Why Real Eigenvalues?

Let $Ax = \lambda x$ ($x \neq 0$).

Take complex conjugate:

\[
A \bar{x} = \bar{\lambda} \bar{x}
\]

Multiply first equation by $\bar{x}^T$ (left):

\[
\bar{x}^T A x = \lambda \bar{x}^T x
\]

Transpose:

\[
x^T A^T \bar{x} = \bar{\lambda} x^T \bar{x}
\]

Since $A = A^T$:

\[
x^T A \bar{x} = \bar{\lambda} x^T \bar{x}
\]

But $x^T A \bar{x} = \bar{x}^T A x = \lambda \bar{x}^T x$, so

\[
\lambda \bar{x}^T x = \bar{\lambda} \bar{x}^T x \quad \Rightarrow \quad \lambda = \bar{\lambda}
\]

$\lambda$ is real!

---

### 4. Rank-1 Expansion

\[
A = \sum_{i=1}^n \lambda_i q_i q_i^T
\]

Each term $\lambda_i q_i q_i^T$ is a scaled **projection** onto the direction $q_i$.

Symmetric matrices are linear combinations of perpendicular rank-1 projections.

---

### 5. Positive Definite Matrices

$A$ (symmetric) is **positive definite** if

\[
\boxed{x^T A x > 0 \quad \forall x \neq 0}
\]

**Equivalent conditions**:

- All eigenvalues $\lambda_i > 0$
- All pivots $> 0$ (in elimination)
- All leading principal minors $> 0$
- $\det(A) > 0$ and all submatrix determinants positive
- Cholesky factorization exists: $A = L L^T$ ($L$ lower triangular)

---

### 6. Example

\[
A = \begin{pmatrix} 5 & 2 \\ 2 & 3 \end{pmatrix}
\]

Eigenvalues: solve $\det(A - \lambda I) = (5-\lambda)(3-\lambda) - 4 = \lambda^2 - 8\lambda + 11 = 0$

\[
\lambda = \frac{8 \pm \sqrt{64-44}}{2} = 4 \pm \sqrt{5} \quad (\text{both positive})
\]

Quadratic form:

\[
x^T A x = 5x_1^2 + 4x_1 x_2 + 3x_2^2 = (x_1 + x_2)^2 + 4x_1^2 + 2x_2^2 > 0 \quad (x \neq 0)
\]

Positive definite.

---

### Code: Check Positive Definiteness

```{code-cell} python
import numpy as np
from IPython.display import Markdown, display

A = np.array([[5, 2],
              [2, 3]], dtype=float)

eigvals = np.linalg.eigvals(A)
is_pd_eig = np.all(eigvals > 0)

# Cholesky (fails if not PD)
try:
    L = np.linalg.cholesky(A)
    cholesky_success = True
except np.linalg.LinAlgError:
    cholesky_success = False

display(Markdown(f"""
**Matrix** $$ A = \\begin{{pmatrix}} 5 & 2 \\\\ 2 & 3 \\end{{pmatrix}} $$

**Eigenvalues**: $$ {np.round(eigvals, 4).tolist()} $$

Positive definite (eigenvalues)? **{'Yes' if is_pd_eig else 'No'}**

Cholesky factorization exists? **{'Yes' if cholesky_success else 'No'}**

$$ L = \\begin{{pmatrix}} {L[0,0]:.3f} & 0 \\\\ {L[1,0]:.3f} & {L[1,1]:.3f} \\end{{pmatrix}} $$ (if successful)
"""))
```

## Lecture 26: Complex matrices; fast fourier transform

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/M0Sa8fLOajA"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

We extend linear algebra to complex numbers — necessary for the **Fourier transform** and its fast algorithm, the **FFT**.

---

### 1. Complex Inner Product

For complex vectors $x, y \in \mathbb{C}^n$:

\[
\boxed{\langle y, x \rangle = y^H x}
\]

where $y^H = \bar{y}^T$ is the **conjugate transpose** (Hermitian transpose).

- $\|x\|^2 = x^H x = \sum |x_i|^2 \geq 0$
- Orthogonality: $y^H x = 0$

---

### 2. Hermitian Matrices

A complex matrix $A$ is **Hermitian** if

\[
\boxed{A^H = A}
\]

Properties (analogous to real symmetric):

- Diagonal entries are real.
- Eigenvalues are **real**.
- Eigenvectors for distinct eigenvalues are orthogonal (w.r.t. complex inner product).
- Orthonormal basis of eigenvectors exists: $Q$ with $Q^H Q = I$.

Then:

\[
\boxed{A = Q \Lambda Q^H}, \quad \Lambda \text{ real diagonal}
\]

Such $Q$ is called **unitary** ($Q^{-1} = Q^H$).

---

### 3. Example: 2×2 Hermitian

\[
A = \begin{pmatrix} 2 & 3+i \\ 3-i & 5 \end{pmatrix}
\]

$A^H = A$, diagonal real → Hermitian.

Eigenvalues real, eigenvectors orthogonal under $y^H x = 0$.

---

### 4. Fourier Matrix

The $n \times n$ **Fourier matrix** $F_n$:

\[
(F_n)_{jk} = \omega^{jk}, \quad \omega = e^{2\pi i / n} = \cos\frac{2\pi}{n} + i \sin\frac{2\pi}{n}
\]

(Indices usually $j,k = 0,\dots,n-1$.)

Example $n=4$ ($\omega = i$):

\[
F_4 = \begin{pmatrix}
1 & 1 & 1 & 1 \\
1 & i & i^2 & i^3 \\
1 & i^2 & i^4 & i^6 \\
1 & i^3 & i^6 & i^9
\end{pmatrix}
= \begin{pmatrix}
1 & 1 & 1 & 1 \\
1 & i & -1 & -i \\
1 & -1 & 1 & -1 \\
1 & -i & -1 & i
\end{pmatrix}
\]

Key property: columns are orthogonal under complex inner product.

Scaled version is unitary: $F_n^H F_n = n I$.

---

### 5. Fast Fourier Transform (FFT)

Direct matrix-vector multiply $F_n x$: $O(n^2)$ operations.

**Cooley–Tukey FFT** exploits structure recursively:

\[
F_{2m} = P \begin{pmatrix} I & D \\ I & -D \end{pmatrix} \begin{pmatrix} F_m & 0 \\ 0 & F_m \end{pmatrix}
\]

- $D$: diagonal with powers of $\omega^2$
- $P$: permutation (evens first, odds second)

Cost recurrence:

\[
T(n) = 2 T(n/2) + O(n) \quad \Rightarrow \quad \boxed{T(n) = O(n \log n)}
\]

Huge speedup: $n=10^6$ → $10^{12}$ vs $\sim 2 \times 10^7$ operations.

---

### Code: Fourier Matrix and FFT Comparison

```{code-cell} python
import numpy as np
from IPython.display import Markdown, display

n = 8
omega = np.exp(2j * np.pi / n)
j, k = np.meshgrid(np.arange(n), np.arange(n))
F = omega ** (j * k)

# Orthonormality check (up to scaling)
scaled_F = F / np.sqrt(n)
error = np.linalg.norm(scaled_F.conj().T @ scaled_F - np.eye(n))

# Compare direct vs np.fft
x = np.random.randn(n) + 1j * np.random.randn(n)
direct = F @ x
fft_result = np.fft.fft(x)

display(Markdown(f"""
**Fourier matrix** $$ F_{{{n}}} $$ (first 4×4 block shown):

$$  \\begin{{pmatrix}}
{np.real(F[:4,:4]).astype(int)} + i\\{np.imag(F[:4,:4]).astype(int)}
\\end{{pmatrix}}  $$

Scaled $$ F/\\sqrt{{{n}}} $$ is unitary (error $$ \\approx {error:.2e} $$)

**Direct multiply**: {np.round(direct[:4], 3)}

**np.fft.fft**      : {np.round(fft_result[:4], 3)}

Match? **{'Yes' if np.allclose(direct, fft_result) else 'No'}**
"""))
```

## Lecture 27: Positive definite matrices and minima

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/vF7eyJ2g3kU"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

Positive definite matrices define **quadratic functions with a unique global minimum** — the foundation of optimization and least squares.

---

### 1. Tests for Positive Definiteness (2×2 Case)

For symmetric

\[
A = \begin{pmatrix} a & b \\ b & c \end{pmatrix}
\]

$A$ is **positive definite** $\iff$

- All eigenvalues $> 0$  
- $a > 0$ and $\det(A) = ac - b^2 > 0$  
- Both pivots $> 0$: $a > 0$, $\frac{ac-b^2}{a} > 0$  
- $\boxed{x^T A x > 0 \quad \forall x \neq 0}$

**Positive semidefinite**: $x^T A x \geq 0$ (allows zero eigenvalue, $\det(A) = 0$).

---

### 2. Examples

| Matrix | $x^T A x$ | Classification | Reason |
|-------|-----------|----------------|-------|
| $\begin{pmatrix} 2 & 6 \\ 6 & 18 \end{pmatrix}$ | $2x_1^2 + 12x_1 x_2 + 18x_2^2 = 2(x_1 + 3x_2)^2 \geq 0$ | Semidefinite | $\det=0$, rank 1 |
| $\begin{pmatrix} 2 & 6 \\ 6 & 7 \end{pmatrix}$ | $2x_1^2 + 12x_1 x_2 + 7x_2^2$ | Indefinite | Saddle (can be negative) |
| $\begin{pmatrix} 2 & 6 \\ 6 & 20 \end{pmatrix}$ | $2x_1^2 + 12x_1 x_2 + 20x_2^2 = 2(x_1 + 3x_2)^2 + 2x_2^2 > 0$ | Definite | Completed square positive |

---

### 3. Completing the Square = Gaussian Elimination

For $f(x,y) = ax^2 + 2hxy + cy^2$:

\[
f = a\left(x + \frac{h}{a}y\right)^2 + \left(c - \frac{h^2}{a}\right)y^2
\]

- First coefficient $a$ → first pivot  
- Second coefficient $c - h^2/a$ → second pivot

Positive definite $\iff$ both coefficients $> 0$.

---

### 4. Multivariable Case: Second Derivative Test

Consider $f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^T A \mathbf{x} - \mathbf{b}^T \mathbf{x} + c$.

Critical point: $\nabla f = A \mathbf{x} - \mathbf{b} = 0 \quad \Rightarrow \quad \mathbf{x}^* = A^{-1} \mathbf{b}$ (if $A$ invertible).

Second derivative (Hessian): $A$.

- $A$ positive definite $\Rightarrow$ **global minimum** at $\mathbf{x}^*$.
- $A$ indefinite $\Rightarrow$ saddle.

---

### 5. 3×3 Tridiagonal Example

\[
A = \begin{pmatrix}
2 & -1 & 0 \\
-1 & 2 & -1 \\
0 & -1 & 2
\end{pmatrix}
\]

Leading minors: $2 > 0$, $\det\begin{pmatrix}2 & -1\\-1 & 2\end{pmatrix}=3 > 0$, $\det(A)=4 > 0$.

Pivots (after elimination): $2$, $3/2$, $4/3$ (all positive).

Eigenvalues: $2 - \sqrt{2} \approx 0.586$, $2$, $2 + \sqrt{2} \approx 3.414$ (all positive).

Quadratic form contours: **ellipsoids** (not hyperbolas).

Axis lengths $\propto 1/\sqrt{\lambda_i}$.

---

### Code: Visualize Quadratic Forms and Ellipsoids

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

def plot_quadratic_form(A, title):
    x = np.linspace(-3, 3, 400)
    y = np.linspace(-3, 3, 400)
    X, Y = np.meshgrid(x, y)
    Z = A[0,0]*X**2 + 2*A[0,1]*X*Y + A[1,1]*Y**2
    
    plt.figure(figsize=(6,6))
    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label='$$ f(x,y) $$')
    plt.contour(X, Y, Z, levels=[0], colors='red', linewidths=2)
    plt.xlabel('$$ x $$'); plt.ylabel('$$ y $$')
    plt.title(title)
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.show()

# Positive definite
A_pd = np.array([[2, 6], [6, 20]])
plot_quadratic_form(A_pd, 'Positive Definite: Elliptical Bowl')

# Semidefinite
A_psd = np.array([[2, 6], [6, 18]])
plot_quadratic_form(A_psd, 'Positive Semidefinite: Trough')

# Indefinite
A_ind = np.array([[2, 6], [6, 7]])
plot_quadratic_form(A_ind, 'Indefinite: Saddle')
```

## Lecture 28: Similar matrices and Jordan form

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/TSdXJw83kyA"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

Not every matrix is diagonalizable, but **every** square matrix is similar to a nearly diagonal **Jordan form** — the cleanest representative of its similarity class.

---

### 1. Similar Matrices

Two $n \times n$ matrices $A$ and $B$ are **similar** if there exists invertible $M$ such that

\[
\boxed{B = M^{-1} A M}
\]

Consequences:

- Same eigenvalues (characteristic polynomial $\det(A - \lambda I) = \det(B - \lambda I)$)
- Same trace, determinant, rank
- Eigenvectors related by $M$: if $A x = \lambda x$, then $B (M^{-1} x) = \lambda (M^{-1} x)$

Diagonalizable case: $A = M \Lambda M^{-1}$ → $A$ similar to diagonal $\Lambda$.

---

### 2. Example: Diagonalizable

\[
A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}, \quad
\Lambda = \begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix}
\]

Change-of-basis matrix (easy to invert):

\[
M = \begin{pmatrix} 1 & 4 \\ 0 & 1 \end{pmatrix}
\]

Verify $M^{-1} A M = \Lambda$ (or equivalently $M^T A M$ in some symmetric cases).

All matrices with eigenvalues $3,1$ form a similarity class.

---

### 3. Non-Diagonalizable Case

Repeated eigenvalue but insufficient eigenvectors → cannot diagonalize.

Example:

\[
A = \begin{pmatrix} 4 & 1 \\ 0 & 4 \end{pmatrix}
\]

Eigenvalues: $4,4$ (trace $8$, det $16$).

Only one eigenvector $\begin{pmatrix} 1 \\ 0 \end{pmatrix}$ → geometric multiplicity $1 < 2$.

Still similar to itself (trivial), but the **Jordan form** reveals the structure.

---

### 4. Jordan Blocks and Jordan Form

A **Jordan block** $J_k(\lambda)$ ($k \times k$):

\[
J_k(\lambda) = \begin{pmatrix}
\lambda & 1 & & \\
& \lambda & 1 & \\
& & \ddots & 1 \\
& & & \lambda
\end{pmatrix}
\]

**Jordan canonical form theorem**:

Every square matrix $A$ is similar to a block-diagonal **Jordan matrix** $J$:

\[
\boxed{A = P J P^{-1}}
\]

- Diagonal blocks are Jordan blocks.
- Number of blocks for eigenvalue $\lambda$ = geometric multiplicity (dimension of eigenspace).
- Size of largest block = index of eigenvalue.
- Diagonalizable $\iff$ all blocks $1\times1$ (i.e., $J = \Lambda$).

---

### 5. Example: Jordan Form

\[
A = \begin{pmatrix} 4 & 1 \\ 0 & 4 \end{pmatrix} \quad \Rightarrow \quad J = \begin{pmatrix} 4 & 1 \\ 0 & 4 \end{pmatrix}
\]

(Same as $A$ — already in Jordan form.)

Another matrix with same eigenvalues but different structure:

\[
\begin{pmatrix} 5 & 1 \\ -1 & 3 \end{pmatrix}
\]

Trace $8$, det $16$ → same characteristic polynomial → similar.

---

### 6. Zero Eigenvalues Example

\[
A = \begin{pmatrix}
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0
\end{pmatrix}
\]

Eigenvalues: $0$ (multiplicity $4$).

Rank $2$ → nullity $2$ → geometric multiplicity $2$.

Jordan form: two blocks (e.g., one $2\times2$, one $1\times1$, one $1\times1$).

---

### Code: Compute Jordan Form (via Schur or Direct)

```{code-cell} python
import numpy as np
from scipy.linalg import schur, eig
from IPython.display import Markdown, display

# Non-diagonalizable example
A = np.array([[4, 1],
              [0, 4]])

# Real Schur form (triangular, close to Jordan for real matrices)
T, Z = schur(A, output='real')

# Eigenvalues
eigvals = eig(A)[0]

display(Markdown(f"""
**Matrix** $$ A = \\begin{{pmatrix}} 4 & 1 \\\\ 0 & 4 \\end{{pmatrix}} $$

**Eigenvalues**: $$ {np.round(eigvals, 3).tolist()} $$ (repeated)

**Real Schur form** $$ T \\approx $$ Jordan form:

$$  T = \\begin{{pmatrix}} {T[0,0]:.3f} & {T[0,1]:.3f} \\\\ {T[1,0]:.3f} & {T[1,1]:.3f} \\end{{pmatrix}}  $$

Note the $1$ above diagonal → single $2\\times2$ Jordan block for $$ \\lambda=4 $$.

Geometric multiplicity = $1$ (rank$$ (A-4I)=1 $$).
"""))

# Try another matrix
A2 = np.array([[5, 1], [-1, 3]])
T2, Z2 = schur(A2, output='real')
display(Markdown(f"""
**Another similar matrix**:

Schur form confirms same Jordan structure.
"""))
```