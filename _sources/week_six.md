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

# Week 6: Eigenvalues and Eigenvectors

## Lecture 21: Eigenvalues and eigenvectors

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/cdZnhQjJu4I"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

We introduce the most important idea in linear algebra: **eigenvectors** — the directions that are **stretched** (or squashed) by a matrix.

---

### 1. Definition

For a square matrix $A \in \mathbb{R}^{n \times n}$,  
a vector $v \neq 0$ is an **eigenvector** with **eigenvalue** $\lambda$ if  
$$
\boxed{A v = \lambda v}
$$

Rewrite:  
$$
(A - \lambda I) v = 0 \quad \Rightarrow \quad v \in N(A - \lambda I)
$$

So: $\lambda$ is an eigenvalue if $A - \lambda I$ is **singular** → $\det(A - \lambda I) = 0$

---

### 2. Characteristic Polynomial

$$
\boxed{p(\lambda) = \det(A - \lambda I)}
$$

Roots of $p(\lambda) = 0$ → eigenvalues.

---

### 3. Example: Rotation in 2D

$$
A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix} \quad \text{(90° rotation)}
$$

$$
A - \lambda I = \begin{pmatrix} -\lambda & -1 \\ 1 & -\lambda \end{pmatrix}
\quad \Rightarrow \quad
\det = \lambda^2 + 1 = 0 \quad \Rightarrow \quad \lambda = \pm i
$$

**No real eigenvectors** → rotation has no fixed direction.

---

### 4. Example: Scaling + Shear

$$
A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}
$$

$$
\det(A - \lambda I) = (3-\lambda)(2-\lambda) = 0 \quad \Rightarrow \quad \lambda_1 = 3,\; \lambda_2 = 2
$$

For $\lambda = 3$:  
$$
(A - 3I)v = \begin{pmatrix} 0 & 1 \\ 0 & -1 \end{pmatrix} v = 0 \quad \Rightarrow \quad v = \begin{pmatrix} 1 \\ 0 \end{pmatrix}
$$

For $\lambda = 2$:  
$$
v = \begin{pmatrix} 1 \\ 1 \end{pmatrix}
$$

---

### 5. Geometric Intuition

| Eigenvalue | Meaning |
|----------|-------|
| $\lambda > 1$ | Stretch |
| $0 < \lambda < 1$ | Shrink |
| $\lambda = 0$ | Collapse to line |
| $\lambda < 0$ | Flip + scale |

---

### Code: Find Eigenvalues & Eigenvectors

```{code-cell} python
import numpy as np
from IPython.display import Markdown, display

A = np.array([[3, 1],
              [0, 2]])

eigvals, eigvecs = np.linalg.eig(A)

display(Markdown(f"""
**Matrix** $A = {A.tolist()}$

**Eigenvalues**: ${np.round(eigvals, 3).tolist()}$

**Eigenvectors** (columns):  
$$    
\\begin{{pmatrix}}
{eigvecs[0,0]:.3f} & {eigvecs[0,1]:.3f} \\\\
{eigvecs[1,0]:.3f} & {eigvecs[1,1]:.3f}
\\end{{pmatrix}}
    $$

**Check**: $A v_1 = {np.round(A @ eigvecs[:,0], 3).tolist()} \approx {np.round(eigvals[0] * eigvecs[:,0], 3).tolist()}$
"""))
```

## Lecture 22: Diagonalization and powers of A

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/13r9QY6cmjc"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

If $A$ has **n independent eigenvectors**, we can **diagonalize** it — and compute **powers instantly**.

---

### 1. Diagonalization

Let $S$ = matrix of eigenvectors (columns),  
$\Lambda$ = diagonal matrix of eigenvalues.

Then:  

\[
\boxed{A = S \Lambda S^{-1}} \quad \Rightarrow \quad
\boxed{A^k = S \Lambda^k S^{-1}}
\]

**Derivation**:

\[
A S = S \Lambda \quad \Rightarrow \quad S^{-1} A S = \Lambda
\]

---

### 2. Example

\[
A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}, \quad
S = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}, \quad
\Lambda = \begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix}
\]

\[
A^{10} = S \begin{pmatrix} 3^{10} & 0 \\ 0 & 2^{10} \end{pmatrix} S^{-1}
= \begin{pmatrix} 59049 & 59048 \\ 0 & 1024 \end{pmatrix}
\]

---

### 3. When Does Diagonalization Work?

**Theorem**: $A$ is diagonalizable $\iff$  

- It has $n$ eigenvalues (counting multiplicity)  
- And $n$ **independent** eigenvectors

Distinct eigenvalues $\Rightarrow$ automatically independent eigenvectors $\Rightarrow$ diagonalizable.

Repeated eigenvalues: must check (may or may not be diagonalizable).

Example (not diagonalizable):

\[
A = \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}
\]

Double eigenvalue $2$, only one eigenvector $\begin{pmatrix} 1 \\ 0 \end{pmatrix}$.

---

### 4. Powers and Stability

\[
A^k = S \Lambda^k S^{-1} = \sum_{i=1}^n \lambda_i^k \, s_i t_i^T
\]

(outer products of right/left eigenvectors).

**Convergence**:

- $A^k \to 0$ if all $|\lambda_i| < 1$.
- Dominant term: largest $|\lambda_i|^k$.

---

### 5. Difference Equations: Fibonacci

\[
F_{k+2} = F_{k+1} + F_k
\]

State vector:

\[
u_k = \begin{pmatrix} F_{k+1} \\ F_k \end{pmatrix}, \quad
A = \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}
\]

\[
u_k = A^k u_0
\]

Eigenvalues:

\[
\lambda = \frac{1 \pm \sqrt{5}}{2} \quad (\phi \approx 1.618, \, 1-\phi \approx -0.618)
\]

Distinct $\Rightarrow$ diagonalizable.

\[
F_k = c_1 \phi^k + c_2 (1-\phi)^k
\]

$|(1-\phi)| < 1 \Rightarrow$ second term $\to 0$, $F_k \approx c_1 \phi^k$ (round to nearest integer).

---

### Code: Compute Powers via Diagonalization

```{code-cell} python
import numpy as np
from IPython.display import Markdown, display

A = np.array([[3, 1],
              [0, 2]])

eigvals, S = np.linalg.eig(A)
Lambda = np.diag(eigvals)
S_inv = np.linalg.inv(S)

k = 10
A_pow_direct = np.linalg.matrix_power(A, k)

# Via diagonalization
A_pow_diag = S @ np.diag(eigvals**k) @ S_inv

display(Markdown(f"""
**Matrix** $$ A = \\begin{{pmatrix}} 3 & 1 \\\\ 0 & 2 \\end{{pmatrix}} $$

**Eigenvalues** $$ \\lambda = {np.round(eigvals, 3).tolist()} $$

**A^{{{k}}}** (direct):  

$$  \\begin{{pmatrix}} {int(A_pow_direct[0,0])} & {int(A_pow_direct[0,1])} \\\\ {int(A_pow_direct[1,0])} & {int(A_pow_direct[1,1])} \\end{{pmatrix}}  $$

**A^{{{k}}}** (diagonalization): matches exactly!

Try large $$ k $$ (e.g., 100) — direct method overflows, diagonalization stays accurate.
"""))
```

## Lecture 23: Differential equations and exp(At)

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/IZqwi0wJovM"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

Linear systems of differential equations $\dot{u} = A u$ are solved using eigenvalues and eigenvectors — the continuous analogue of discrete powers $A^k$.

---

### 1. System of ODEs

\[
\boxed{\frac{du}{dt} = A u}
\]

Assume solution $u(t) = e^{\lambda t} x$ (exponential growth/decay).

Substitute:

\[
\lambda e^{\lambda t} x = A e^{\lambda t} x \quad \Rightarrow \quad A x = \lambda x
\]

Same eigenvalue problem!

---

### 2. Example System

\[
\frac{du_1}{dt} = -u_1 + 2 u_2, \quad
\frac{du_2}{dt} = u_1 - 2 u_2
\]

\[
A = \begin{pmatrix} -1 & 2 \\ 1 & -2 \end{pmatrix}
\]

Characteristic equation:

\[
\det(A - \lambda I) = \lambda(\lambda + 3) = 0 \quad \Rightarrow \quad \lambda_1 = 0,\; \lambda_2 = -3
\]

Eigenvectors:

- $\lambda=0$: $x_1 = \begin{pmatrix} 2 \\ 1 \end{pmatrix}$
- $\lambda=-3$: $x_2 = \begin{pmatrix} 1 \\ -1 \end{pmatrix}$

General solution:

\[
\boxed{u(t) = c_1 \begin{pmatrix} 2 \\ 1 \end{pmatrix} + c_2 e^{-3t} \begin{pmatrix} 1 \\ -1 \end{pmatrix}}
\]

Interpretation:

- Steady state: eigenvector for $\lambda=0$.
- Transient: $e^{-3t} \to 0$ (stable).

---

### 3. Stability

For $\dot{u} = A u$:

- **Stable** (solutions $\to 0$) if all $\operatorname{Re}(\lambda_i) < 0$.
- **Steady state nonzero** if some $\lambda_i = 0$ and others negative.
- **Unstable** (blow-up) if any $\operatorname{Re}(\lambda_i) > 0$.

Trace $< 0$ and $\det > 0$ $\Rightarrow$ both eigenvalues negative (2×2 real case).

---

### 4. Decoupling via Diagonalization

Let $S$ = eigenvectors, $u_0 = S c$:

\[
u(t) = S e^{\Lambda t} S^{-1} u_0 = S \begin{pmatrix} e^{\lambda_1 t} & \\ & e^{\lambda_2 t} \end{pmatrix} c
\]

Equations in $v = S^{-1} u$ become **uncoupled**:

\[
\dot{v}_i = \lambda_i v_i
\]

---

### 5. Matrix Exponential

Define via Taylor series:

\[
\boxed{e^{At} = \sum_{k=0}^\infty \frac{(At)^k}{k!} = I + At + \frac{(At)^2}{2!} + \cdots}
\]

If $A$ diagonalizable ($A = S \Lambda S^{-1}$):

\[
\boxed{e^{At} = S e^{\Lambda t} S^{-1}}, \quad
e^{\Lambda t} = \begin{pmatrix} e^{\lambda_1 t} &  &  \\  & \ddots &  \\  &  & e^{\lambda_n t} \end{pmatrix}
\]

Solution:

\[
u(t) = e^{At} u(0)
\]

Converges to steady state if all $\operatorname{Re}(\lambda_i) < 0$.

---

### 6. Comparison: Discrete vs Continuous

| Discrete $u_{k+1} = A u_k$ | Continuous $\dot{u} = A u$ |
|---------------------------|----------------------------|
| $u_k = A^k u_0$           | $u(t) = e^{At} u_0$       |
| Stability: $\|\lambda_i\| < 1$ | Stability: $\operatorname{Re}(\lambda_i) < 0$ |

---

### Code: Solve $\dot{x} = Ax$

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

A = np.array([[-1, 2],
              [1, -2]])
t = np.linspace(0, 5, 400)

# Diagonalize
eigvals, S = np.linalg.eig(A)
Lambda = np.diag(eigvals)
S_inv = np.linalg.inv(S)

def u(t, u0):
    return S @ np.diag(np.exp(eigvals * t)) @ S_inv @ u0

# Initial conditions
u0_list = [np.array([3, 0]), np.array([0, 3]), np.array([2, 1])]

plt.figure(figsize=(10, 6))
for u0 in u0_list:
    traj = np.array([u(ti, u0) for ti in t])
    plt.plot(traj[:,0], traj[:,1], label=f'u0 = {u0}')
    plt.plot(u0[0], u0[1], 'o', markersize=8)

plt.axhline(0, color='k', lw=0.5); plt.axvline(0, color='k', lw=0.5)
plt.xlabel('$$ u_1 $$'); plt.ylabel('$$ u_2 $$')
plt.title('Solutions $$ \\dot{u} = A u $$ → steady state along $$ \\lambda=0 $$')
plt.legend(); plt.grid(alpha=0.3)
plt.show()

display(Markdown("""
All trajectories approach the steady-state line (eigenvector for $$ \\lambda=0 $$).

The fast decay direction is the eigenvector for $$ \\lambda=-3 $$.
"""))
```

## Lecture 24: Markov matrices; fourier series

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/lGGDIGizcQ0"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

Two applications: **probability flows** and **signal decomposition**.

---

### 1. Markov Matrices

A **stochastic matrix** $A$ has:
- Non-negative entries  
- Each **column sums to 1**

Then $A^k$ gives **k-step transition probabilities**.

**Theorem**: If $A$ is stochastic and **regular** (some $k$: $A^k > 0$), then  
\[
\lim_{k \to \infty} A^k = \begin{pmatrix} \pi \\ \pi \\ \vdots \\ \pi \end{pmatrix}
\quad \text{(all rows = steady-state $\pi$)}
\]

And $\pi$ is the **eigenvector for $\lambda = 1$** with $\pi^T \mathbf{1} = 1$.

Why $\lambda=1$?  
$A \mathbf{1} = \mathbf{1}$ (columns sum to 1) $\Rightarrow$ right eigenvector with $\lambda=1$.  
Steady state $\pi^T A = \pi^T$ $\Rightarrow$ left eigenvector with $\lambda=1$.

All other eigenvalues satisfy $|\lambda_i| \leq 1$, and for regular matrices $<1$.

---

### 2. Example: 2-State Markov Chain

Transition matrix:

\[
A = \begin{pmatrix} 0.9 & 0.2 \\ 0.1 & 0.8 \end{pmatrix}
\]

Eigenvalues: $1$ and $0.7$.

Eigenvectors:

- $\lambda=1$: $\begin{pmatrix} 2 \\ 1 \end{pmatrix}$ (steady state proportions)
- $\lambda=0.7$: $\begin{pmatrix} -1 \\ 1 \end{pmatrix}$

Initial $u_0 = \begin{pmatrix} 0 \\ 1000 \end{pmatrix}$:

\[
u_k = c_1 \begin{pmatrix} 2 \\ 1 \end{pmatrix} + c_2 (0.7)^k \begin{pmatrix} -1 \\ 1 \end{pmatrix}
\]

$c_1 = 1000/3$, $c_2 = 2000/3$ $\Rightarrow$ transient term decays, approaches $\begin{pmatrix} 2000/3 \\ 1000/3 \end{pmatrix}$.

---

### 3. Fourier Series: Eigenfunctions of $d^2/dx^2$

Operator: $L = \frac{d^2}{dx^2}$ on $[0, 2\pi]$  
Eigenfunctions:  
\[
\boxed{\phi_n(x) = e^{i n x} \quad \Rightarrow \quad L \phi_n = -n^2 \phi_n}
\]

Any function $f(x)$:  
\[
\boxed{f(x) = \sum_{n=-\infty}^\infty c_n e^{i n x}}, \quad
c_n = \frac{1}{2\pi} \int_0^{2\pi} f(x) e^{-i n x} dx
\]

Orthogonality (complex inner product over interval):

\[
\int_0^{2\pi} e^{i m x} e^{-i n x} dx = 2\pi \delta_{mn}
\]

---

### 4. Connection

| Concept | Matrix $A$ | Eigenvalue | Meaning |
|-------|-----------|----------|-------|
| Markov | Stochastic | $\lambda=1$ | Steady state |
| Fourier | $L = d^2/dx^2$ | $\lambda = -n^2$ | Frequency |

Both decompose evolution into independent modes scaled by eigenvalues.

---

### Code: Markov Chain Steady State

```{code-cell} python
import numpy as np
from IPython.display import Markdown, display

A = np.array([[0.9, 0.2],
              [0.1, 0.8]], dtype=float)

# Find eigenvector for lambda=1 (left eigenvector for stochastic)
eigvals, eigvecs = np.linalg.eig(A.T)  # transpose for left
idx = np.argmin(np.abs(eigvals - 1))
pi = np.real(eigvecs[:, idx])
pi = pi / pi.sum()

# Simulate many steps
u0 = np.array([0, 1000])
u100 = np.linalg.matrix_power(A, 100) @ u0

display(Markdown(f"""
**Transition matrix** $$ A $$:

$$    
A = \\begin{{pmatrix}}
0.9 & 0.2 \\\\
0.1 & 0.8
\\end{{pmatrix}}
    $$

**Steady-state** $$ \\pi $$ (eigenvector for $$ \\lambda=1 $$):  
$$    
\\pi = \\begin{{pmatrix}} {pi[0]:.3f} \\\\ {pi[1]:.3f} \\end{{pmatrix}}
    $$

**After 100 steps** from $$ [0, 1000]^T $$:  
$$ {np.round(u100).astype(int).tolist()} $$

**Check**: $$ A \\pi = \\pi $$ → {np.allclose(A @ pi, pi)}$
"""))
```