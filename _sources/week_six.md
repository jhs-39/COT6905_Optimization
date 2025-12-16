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

We introduce the most important idea in linear algebra: **eigenvectors** — directions that are **stretched** (or squashed) by a matrix, but **not rotated**.

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

Let $P$ = matrix of eigenvectors (columns),  
$D$ = diagonal matrix of eigenvalues.

Then:  
$$
\boxed{A = P D P^{-1}} \quad \Rightarrow \quad
\boxed{A^k = P D^k P^{-1}}
$$

---

### 2. Example

$$
A = \begin{pmatrix} 3 & 1 \\ 0 & 2 \end{pmatrix}, \quad
P = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}, \quad
D = \begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix}
$$

$$
A^{10} = P \begin{pmatrix} 3^{10} & 0 \\ 0 & 2^{10} \end{pmatrix} P^{-1}
= \begin{pmatrix} 59049 & 59048 \\ 0 & 1024 \end{pmatrix}
$$

---

### 3. When Does Diagonalization Work?

**Theorem**: $A$ is diagonalizable $\iff$  
- It has $n$ eigenvalues (counting multiplicity)  
- And $n$ **independent** eigenvectors

---

### 4. Fibonacci via Diagonalization

$$
\begin{pmatrix} F_{n+1} \\ F_n \end{pmatrix}
= \begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix}
\begin{pmatrix} F_n \\ F_{n-1} \end{pmatrix}
= A^n \begin{pmatrix} 1 \\ 0 \end{pmatrix}
$$

Diagonalize $A$ → closed form:  
$$
\boxed{F_n = \frac{\phi^n - (1-\phi)^n}{\sqrt{5}}}
\quad \phi = \frac{1+\sqrt{5}}{2}
$$

---

### Code: Compute $A^{100}$ in One Line

```{code-cell} python
import numpy as np
from IPython.display import Markdown, display

A = np.array([[3, 1],
              [0, 2]])

P, D = np.linalg.eig(A)
D_mat = np.diag(D)
k = 100
Ak = P @ np.diag(D**k) @ np.linalg.inv(P)

display(Markdown(f"""
$A^{{{k}}} = \\begin{{pmatrix}} {int(Ak[0,0])} & {int(Ak[0,1])} \\\\ {int(Ak[1,0])} & {int(Ak[1,1])} \\end{{pmatrix}}$
"""))
```

## Lecture 23: Differential equations and exp(At)

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/IZqwi0wJovM"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

Matrix exponentials solve **systems of ODEs**:  
$$
\frac{dx}{dt} = A x \quad \Rightarrow \quad
\boxed{x(t) = e^{At} x(0)}
$$

---

### 1. Matrix Exponential

$$
\boxed{e^{At} = \sum_{k=0}^\infty \frac{(At)^k}{k!}}
= I + At + \frac{(At)^2}{2!} + \cdots
$$

If $A = P D P^{-1}$, then  
$$
\boxed{e^{At} = P e^{Dt} P^{-1}}, \quad
e^{Dt} = \begin{pmatrix} e^{\lambda_1 t} &  &  \\  & \ddots &  \\  &  & e^{\lambda_n t} \end{pmatrix}
$$

---

### 2. Example: Damped Oscillator

$$
\frac{d^2 y}{dt^2} + 2\frac{dy}{dt} + 2y = 0
\quad \Rightarrow \quad
x = \begin{pmatrix} y \\ y' \end{pmatrix}, \;
A = \begin{pmatrix} 0 & 1 \\ -2 & -2 \end{pmatrix}
$$

Eigenvalues: $\lambda = -1 \pm i$ → solution:  
$$
y(t) = e^{-t} (c_1 \cos t + c_2 \sin t)
$$

---

### 3. Stability

| Eigenvalues | Behavior of $e^{At}$ |
|-----------|------------------------|
| $\text{Re}(\lambda) < 0$ | $\to 0$ (stable) |
| $\text{Re}(\lambda) > 0$ | $\to \infty$ (unstable) |
| $\text{Re}(\lambda) = 0$ | Oscillations |

---

### Code: Solve $\dot{x} = Ax$

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

A = np.array([[0, 1],
              [-2, -2]])
t = np.linspace(0, 10, 400)

# Diagonalize
P, D = np.linalg.eig(A)
D = np.diag(D)

def x(t, x0):
    return P @ (np.exp(D * t) @ np.linalg.solve(P, x0))

x0 = np.array([1, 0])
traj = np.array([x(ti, x0) for ti in t])

plt.figure(figsize=(10, 5))
plt.plot(t, traj[:,0], label='$y(t)$')
plt.plot(t, traj[:,1], label="$y'(t)$")
plt.axhline(0, color='k', linewidth=0.5)
plt.xlabel('Time $t$'); plt.title('Solution: $e^{At} x(0)$')
plt.legend(); plt.grid(alpha=0.3)
plt.show()
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
$$
\lim_{k \to \infty} A^k = \begin{pmatrix} \pi \\ \pi \\ \vdots \\ \pi \end{pmatrix}
\quad \text{(all rows = steady-state $\pi$)}
$$

And $\pi$ is the **eigenvector for $\lambda = 1$** with $\pi^T \mathbf{1} = 1$.

---

### 2. Example: Google PageRank (simplified)

$$
A = \begin{pmatrix}
0 & 1/2 & 1/2 \\
1/3 & 0 & 1/2 \\
2/3 & 1/2 & 0
\end{pmatrix}
$$

$\lambda = 1$ → eigenvector $\pi$ → ranking.

---

### 3. Fourier Series: Eigenfunctions of $d^2/dx^2$

Operator: $L = \frac{d^2}{dx^2}$ on $[0, 2\pi]$  
Eigenfunctions:  
$$
\boxed{\phi_n(x) = e^{i n x} \quad \Rightarrow \quad L \phi_n = -n^2 \phi_n}
$$

Any function $f(x)$:  
$$
\boxed{f(x) = \sum_{n=-\infty}^\infty c_n e^{i n x}}, \quad
c_n = \frac{1}{2\pi} \int_0^{2\pi} f(x) e^{-i n x} dx
$$

---

### 4. Connection

| Concept | Matrix $A$ | Eigenvalue | Meaning |
|-------|-----------|----------|-------|
| Markov | Stochastic | $\lambda=1$ | Steady state |
| Fourier | $L = d^2/dx^2$ | $\lambda = -n^2$ | Frequency |

---

### Code: Markov Chain Steady State

```{code-cell} python
import numpy as np
from IPython.display import Markdown, display

A = np.array([[0, 0.5, 0.5],
              [1/3, 0, 0.5],
              [2/3, 0.5, 0]], dtype=float)

# Find eigenvector for lambda=1
eigvals, eigvecs = np.linalg.eig(A.T)  # left eigenvector
pi = eigvecs[:, np.isclose(eigvals, 1)].real
pi = pi / pi.sum()

display(Markdown(f"""
**Transition matrix** $A$:

$$  
A = \\begin{{pmatrix}}
0 & 0.5 & 0.5 \\\\
1/3 & 0 & 0.5 \\\\
2/3 & 0.5 & 0
\\end{{pmatrix}}
  $$

**Steady-state** $\pi$ (eigenvector for $\lambda=1$):  
$$  
\\pi = \\begin{{pmatrix}} {pi[0]:.3f} \\\\ {pi[1]:.3f} \\\\ {pi[2]:.3f} \\end{{pmatrix}}
  $$

**Check**: $A \\pi = \\pi$ → {np.allclose(A @ pi, pi)}$
"""))
```