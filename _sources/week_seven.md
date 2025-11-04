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

Symmetric matrices have **real eigenvalues** and **orthogonal eigenvectors** — the foundation of quadratic forms and optimization.

---

### 1. Symmetric Matrix

$A$ is **symmetric** if  
$$
\boxed{A = A^T}
$$

---

### 2. Spectral Theorem (Real Case)

If $A \in \mathbb{R}^{n \times n}$ is symmetric, then:  
$$
\boxed{A = Q \Lambda Q^T}
$$
where  
- $Q$ is **orthogonal** ($Q^T Q = I$)  
- $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_n)$  
- All $\lambda_i$ are **real**

---

### 3. Positive Definite Matrices

$A$ is **positive definite** if  
$$
\boxed{x^T A x > 0 \quad \forall x \neq 0}
$$

**Equivalent conditions**:  
- All eigenvalues $\lambda_i > 0$  
- All leading minors $> 0$  
- $A = LL^T$ (Cholesky)

---

### 4. Example

$$
A = \begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix}
$$

Eigenvalues: $\lambda = 1, 3$ → positive definite.

Quadratic form:  
$$
x^T A x = 2x_1^2 - 2x_1 x_2 + 2x_2^2 = (x_1 - x_2)^2 + x_1^2 + x_2^2 > 0
$$

---

### Code: Check Positive Definiteness

```{code-cell} python
import numpy as np
from IPython.display import Markdown, display

A = np.array([[2, -1],
              [-1, 2]])

eigvals = np.linalg.eigvals(A)
is_pd = np.all(eigvals > 0)

display(Markdown(f"""
**Matrix** $A = \\begin{{pmatrix}} 2 & -1 \\\\ -1 & 2 \\end{{pmatrix}}$

**Eigenvalues**: ${np.round(eigvals, 3).tolist()}$

$A$ is **{'positive definite' if is_pd else 'not positive definite'}**

**Cholesky**: $A = LL^T$ where  
$L = \\begin{{pmatrix}} {np.sqrt(2):.3f} & 0 \\\\ {-0.5/np.sqrt(2):.3f} & {np.sqrt(1.5):.3f} \\end{{pmatrix}}$
"""))
```

## Lecture 26: Complex matrices; fast fourier transform

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/M0Sa8fLOajA"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

Complex eigenvalues come in pairs — and power the **FFT**, the fastest way to compute the DFT.

---

### 1. Complex Eigenvalues

For real $A$, if $\lambda = a + bi$ is an eigenvalue, so is $\bar{\lambda} = a - bi$.

Eigenvectors are complex, but solutions stay real via Euler:  
$$
e^{(a+bi)t} = e^{at} (\cos(bt) + i \sin(bt))
$$

---

### 2. Discrete Fourier Transform (DFT)

Given $x_0, \dots, x_{n-1}$,  
$$
\boxed{X_k = \sum_{j=0}^{n-1} x_j \omega^{jk}}, \quad \omega = e^{-2\pi i / n}
$$

Matrix form: $X = F x$, where $F_{jk} = \omega^{jk}$

---

### 3. Fast Fourier Transform (FFT)

**Cooley-Tukey**: Split into even/odd →  
$$
\boxed{\text{DFT of size } n \text{ in } O(n \log n) \text{ time}}
$$

---

### 4. Example: $n=4$

$$
F_4 = \begin{pmatrix}
1 & 1 & 1 & 1 \\
1 & -i & -1 & i \\
1 & -1 & 1 & -1 \\
1 & i & -1 & -i
\end{pmatrix}
$$

---

### Code: FFT vs DFT

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

# Signal
n = 8
t = np.linspace(0, 1, n, endpoint=False)
x = np.sin(2*np.pi*2*t) + 0.5*np.sin(2*np.pi*5*t)

X_fft = np.fft.fft(x)
X_dft = np.array([sum(x[j] * np.exp(-2j*np.pi*k*j/n) for j in range(n)) for k in range(n)])

display(Markdown(f"""
**Signal** $x$: {np.round(x, 3).tolist()}

**FFT result**: {np.round(X_fft, 3).tolist()}

**DFT result**: {np.round(X_dft, 3).tolist()}

**Match?** {np.allclose(X_fft, X_dft)}
"""))

plt.figure(figsize=(10,4))
plt.stem(np.abs(X_fft))
plt.title('FFT Magnitude')
plt.xlabel('Frequency bin')
plt.show()
```

## Lecture 27: Positive definite matrices and minima

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/vF7eyJ2g3kU"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

Positive definite matrices define **convex quadratic functions** — with **unique global minima**.

---

### 1. Quadratic Form

$$
f(x) = \frac{1}{2} x^T A x - b^T x + c
$$

---

### 2. Critical Point

$$
\nabla f = A x - b = 0 \quad \Rightarrow \quad
\boxed{x^* = A^{-1} b}
$$

---

### 3. Second Derivative Test

Hessian = $A$.  
If $A$ is **positive definite**, then $x^*$ is a **global minimum**.

---

### 4. Example: Least Squares

$$
\min_x \|Ax - b\|^2 = \min_x \left( x^T (A^T A) x - 2 b^T A x + b^T b \right)
$$

Hessian = $A^T A$ → positive definite if $A$ has independent columns → **unique solution**.

---

### 5. Geometry

| $A$ | Bowl Shape |
|-----|-----------|
| Positive definite | Elliptical bowl (minimum) |
| Positive semidefinite | Trough (minimum along line) |
| Indefinite | Saddle (no min/max) |

---

### Code: Minimize $f(x) = \frac{1}{2}x^T A x - b^T x$

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

A = np.array([[3, 1],
              [1, 2]])
b = np.array([1, 2])

x_star = np.linalg.solve(A, b)

display(Markdown(f"""
$f(x) = \\frac{{1}}{{2}} x^T A x - b^T x$

$A = \\begin{{pmatrix}} 3 & 1 \\\\ 1 & 2 \\end{{pmatrix}}$, $b = \\begin{{pmatrix}} 1 \\\\ 2 \\end{{pmatrix}}$

**Minimum at** $x^* = \\begin{{pmatrix}} {x_star[0]:.3f} \\\\ {x_star[1]:.3f} \\end{{pmatrix}}$

**Value**: $f(x^*) = {0.5*x_star.T @ A @ x_star - b.T @ x_star:.3f}$
"""))

# Contour plot
x = np.linspace(-1, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = 0.5*(A[0,0]*X**2 + 2*A[0,1]*X*Y + A[1,1]*Y**2) - b[0]*X - b[1]*Y

plt.figure(figsize=(8,6))
plt.contour(X, Y, Z, 20)
plt.plot(x_star[0], x_star[1], 'r*', markersize=15, label='Minimum')
plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
plt.title('Quadratic Function (Positive Definite)')
plt.legend(); plt.grid(alpha=0.3)
plt.show()
```

## Lecture 28: Similar matrices and Jordan form

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/TSdXJw83kyA"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

Not all matrices are diagonalizable — but **every** matrix has a **Jordan form**.

---

### 1. Similar Matrices

$A$ and $B$ are **similar** if  
$$
\boxed{B = P^{-1} A P}
$$

→ Same eigenvalues, same characteristic polynomial.

---

### 2. Jordan Block

For eigenvalue $\lambda$, a **Jordan block** is  
$$
J_k(\lambda) = \begin{pmatrix}
\lambda & 1 &  &  \\
& \lambda & 1 &  \\
&  & \ddots & 1 \\
&  &  & \lambda
\end{pmatrix}
$$

---

### 3. Jordan Canonical Form

**Theorem**: Every square matrix $A$ is similar to a **block diagonal** matrix of Jordan blocks:  
$$
\boxed{A = P J P^{-1}}
$$

- Diagonalizable $\iff$ all blocks are $1\times1$  
- Geometric multiplicity = # of Jordan blocks for $\lambda$  
- Algebraic multiplicity = total size of blocks for $\lambda$

---

### 4. Example

$$
A = \begin{pmatrix} 3 & 1 \\ 0 & 3 \end{pmatrix}
\quad \Rightarrow \quad
J = \begin{pmatrix} 3 & 1 \\ 0 & 3 \end{pmatrix}
\quad (\text{not diagonalizable})
$$

Only one eigenvector: $\begin{pmatrix} 1 \\ 0 \end{pmatrix}$

---

### Code: Compute Jordan Form (via Schur)

```{code-cell} python
import numpy as np
from scipy.linalg import schur
from IPython.display import Markdown, display

A = np.array([[3, 1],
              [0, 3]])

T, Z = schur(A, output='real')  # Real Schur form (close to Jordan)

display(Markdown(f"""
**Matrix** $A = \\begin{{pmatrix}} 3 & 1 \\\\ 0 & 3 \\end{{pmatrix}}$

**Schur form** $T = Z^T A Z \approx$

$$  
\\begin{{pmatrix}} {T[0,0]:.3f} & {T[0,1]:.3f} \\\\ {T[1,0]:.3f} & {T[1,1]:.3f} \\end{{pmatrix}}
  $$

**Jordan block** for $\lambda = 3$ (size 2)

**Check**: $A - 3I = \\begin{{pmatrix}} 0 & 1 \\\\ 0 & 0 \\end{{pmatrix}}$ → rank 1 → geometric multiplicity = 1
"""))
```