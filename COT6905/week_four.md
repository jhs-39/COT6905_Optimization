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

# Week 4: Orthogonal Basis and Projection

## Lecture 13: Concept Review & Consolidation

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/l88D4r74gtM"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

We've include this for your interest and completeness, but feel free to skip if you are confident in your understanding so far; this is a skill/concept review lecture.

## Lecture 14: Orthogonal Basis

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/YzZUIYRCE38"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

We introduce orthogonal vectors, orthogonal subspaces, and orthogonal bases, key for understanding geometric relationships in vector spaces and solving problems like projections. We connect these to the fundamental subspaces from Lecture 10.

We show **why** the row space is orthogonal to the null space, and the column space is orthogonal to the left-null space.  
These two *orthogonal-complement* pairs completely carve $\mathbb{R}^n$ and $\mathbb{R}^m$ into perpendicular pieces.

### 1. Orthogonal Vectors – the dot-product test
Two vectors $u, v \in \mathbb{R}^k$ are **orthogonal** if  
$
u^{T}v = 0 .
$
Geometrically they are perpendicular (90°).  
The Pythagorean theorem holds **only** when the vectors are orthogonal:

$$
\|u+v\|^{2}= (u+v)^{T}(u+v)= u^{T}u + 2u^{T}v + v^{T}v .
\qquad \text{If } u^{T}v=0 \;\Rightarrow\; \|u+v\|^{2}= \|u\|^{2}+\|v\|^{2}.
$$

### 2. Orthogonal Subspaces
Subspaces $S, T \subset \mathbb{R}^k$ are **orthogonal** if **every** vector in $S$ is orthogonal to **every** vector in $T$.  
Notation: $S \perp T$.

### 3. The Four Subspaces are Orthogonal Pairs  

| Pair (in $\mathbb{R}^n$) | Pair (in $\mathbb{R}^m$) |
|--------------------------|--------------------------|
| **Row space** $C(A^{T})$ $\perp$ **Null space** $N(A)$ | **Column space** $C(A)$ $\perp$ **Left-null space** $N(A^{T})$ |

#### Why does $C(A^{T}) \perp N(A)$?
* $x \in N(A)$ means $Ax = 0$.  
* Examine product Ax as the dot product of each row vector in A with x:  
  $
  \text{(row}_i \text{ of }A) \cdot x = 0 \quad \text{for every } i.
  $
* Every row of $A$ (i.e. every vector in the row space) is orthogonal to $x$.  
* Because this holds for **all** $x\in N(A)$, the whole row space is orthogonal to the whole null space.

The same argument (applied to $A^{T}$) shows $C(A) \perp N(A^{T})$.

### 4. Orthogonal Complements  
For a subspace $S\subset \mathbb{R}^k$ its **orthogonal complement** contains all the other orthogonal dimensions in the subspace:
$
\dim S + \dim S^{\perp} = k
$

$
S^{\perp}= \{ v\in\mathbb{R}^k \mid v^{T}u=0 \;\forall u\in S\}
$

Below, we give you a sneak peek of a method to calculate the 4 subspaces using singular value decomposition! We take the dot product to demonstrate orthogonality. It should be 0 or very close to 0 (numerical stability)

```{code-cell} python
import numpy as np
from numpy.linalg import svd

# Example matrix A (not square)
A = np.array([
    [1, 2, 3],
    [4, 5, 6]
], dtype=float)

# --- Compute the four fundamental subspaces via SVD ---
U, s, Vt = svd(A)
r = np.sum(s > 1e-10)   # rank

# Bases for each subspace
row_space = Vt[:r].T              # C(A^T)
null_space = Vt[r:].T             # N(A)
col_space = U[:, :r]              # C(A)
left_null_space = U[:, r:]        # N(A^T)

# --- Check orthogonality numerically ---
def check_orthogonal(X, Y, nameX, nameY):
    dot_product = X.T @ Y
    print(f"{nameX} ⟂ {nameY}?  ||dot|| = {np.linalg.norm(dot_product):.2e}")

check_orthogonal(row_space, null_space, "Row space", "Null space")
check_orthogonal(col_space, left_null_space, "Column space", "Left-null space")

# --- Dimensional sanity checks ---
print("\nDimensional checks:")
print(f"dim(Row space) + dim(Null space) = {row_space.shape[1]} + {null_space.shape[1]} = {A.shape[1]}")
print(f"dim(Column space) + dim(Left-null space) = {col_space.shape[1]} + {left_null_space.shape[1]} = {A.shape[0]}")
```

## Lecture 15: Projections onto subspaces

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/Y_Ac6KiQ1t0"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

We learn how to **find the closest point** in a subspace to a given vector $b$.  
This is the foundation of **least squares** — solving $Ax = b$ when no exact solution exists.

---

### 1. Projecting a Vector onto a Line (2D Intuition)

Suppose we have two vectors in $\mathbb{R}^2$:  
- $a$: defines a **line** through the origin  
- $b$: any point we want to project

**Goal**: Find the point $p$ on the line through $a$ that is **closest** to $b$.

Let $p = x a$ (some scalar multiple of $a$).  
The **error vector** is  
$$
e = b - p = b - x a
$$

For $p$ to be the *closest* point, $e$ must be **perpendicular** to the line (i.e., to $a$):  
$$
a^T e = 0 \quad \Rightarrow \quad a^T (b - x a) = 0
$$
$$
a^T b - x (a^T a) = 0 \quad \Rightarrow \quad x = \frac{a^T b}{a^T a}
$$
$$
\boxed{p = a \left( \frac{a^T b}{a^T a} \right) = \frac{a a^T}{a^T a} \, b}
$$

Let  
$$
\boxed{P = \frac{a a^T}{a^T a}}
\qquad \text{then} \quad
p = P b
$$

$P$ is the **projection matrix** onto the line through $a$.

---

### 2. Properties of the Projection Matrix $P$

| Property | Why it holds |
|--------|--------------|
| $P^2 = P$ | Projecting twice gives the same result |
| $P^T = P$ | Symmetric (follows from $a a^T = (a a^T)^T$) |
| $\text{rank}(P) = 1$ | Column space is the line spanned by $a$ |

---

### 3. General Case: Projecting onto a Subspace $C(A)$

Now let $A$ be an $m \times n$ matrix (not necessarily square).  
We want to project $b \in \mathbb{R}^m$ onto **$C(A)$** — the column space of $A$.

Let  
$$
p = A \hat{x} \quad \text{(so $p \in C(A)$)}
$$
Error:  
$$
e = b - p = b - A \hat{x}
$$

For $p$ to be the **closest point**, $e$ must be **perpendicular to every column of $A$**:  
$$
a_i^T e = 0 \quad \forall i \quad \Rightarrow \quad A^T e = 0
$$
$$
A^T (b - A \hat{x}) = 0 \quad \Rightarrow \quad A^T A \hat{x} = A^T b
$$

If $A^T A$ is invertible (i.e., $A$ has **full column rank**),  
$$
\boxed{\hat{x} = (A^T A)^{-1} A^T b}
$$
$$
\boxed{p = A \hat{x} = A (A^T A)^{-1} A^T b}
$$

Define the **projection matrix onto $C(A)$**:  
$$
\boxed{P = A (A^T A)^{-1} A^T}
$$

---

### 4. Key Properties (Same as Before!)

- $P^2 = P$  
- $P^T = P$  
- $\text{rank}(P) = \text{rank}(A)$  
- $Pb \in C(A)$  
- $b - Pb \in N(A^T)$ (i.e., perpendicular to $C(A)$)

> **Note**: $A$ is **not square** → $P \neq I$, even if $A$ is invertible in some sense.  
> Only when $C(A) = \mathbb{R}^m$ (full row rank) do we get $P = I$.

---

### 5. Example: Project onto the $xy$-Plane

Let  
$$
A = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{pmatrix}, \quad
b = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}
$$

Column space: the $xy$-plane in $\mathbb{R}^3$.

Compute:  
$$
A^T A = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}, \quad
(A^T A)^{-1} = I_2
$$
$$
P = A (A^T A)^{-1} A^T = A A^T = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}
$$
$$
p = P b = \begin{pmatrix} 1 \\ 2 \\ 0 \end{pmatrix}
$$
Error:  
$$
e = b - p = \begin{pmatrix} 0 \\ 0 \\ 3 \end{pmatrix} \in N(A^T)
$$

Check: $e \perp C(A)$?  
$$
\begin{pmatrix} 1 & 0 & 0 \end{pmatrix} \cdot e = 0, \quad
\begin{pmatrix} 0 & 1 & 0 \end{pmatrix} \cdot e = 0
\;\; \checkmark
$$

---

### 6. Why Do We Care? → **Least Squares**

The system $Ax = b$ may have **no solution** (if $b \notin C(A)$).  
Instead, solve the **closest problem**:  
$$
\text{minimize } \|Ax - b\|^2
\quad \Rightarrow \quad
\text{solution: } \hat{x} = (A^T A)^{-1} A^T b
$$

This is **linear regression**, **data fitting**, **curve fitting**, etc.

**Example**: Fit a line $y = c + dt$ to noisy points $(t_i, b_i)$  
→ Let $A = \begin{pmatrix} 1 & t_1 \\ 1 & t_2 \\ \vdots \end{pmatrix}$  
→ Solve $A \hat{x} = P b$ → best-fit line.

---

### Summary: The Big Picture

| Concept | Formula | Meaning |
|-------|--------|--------|
| Projection onto line | $p = \frac{a a^T}{a^T a} b$ | Shadow of $b$ on line |
| Projection onto $C(A)$ | $p = A (A^T A)^{-1} A^T b$ | Shadow of $b$ on column space |
| Error $e = b - p$ | $e \in N(A^T)$ | Perpendicular to subspace |
| Least Squares | $\hat{x} = (A^T A)^{-1} A^T b$ | Best approximate solution |

### Visualization: Projection on the plane

```{code-cell} python
# ------------------------------------------------------------------
#  Lecture 15 – Projection onto a random plane (no widgets)
# ------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import matplotlib as mpl
from IPython.display import Markdown, display

# ------------------------------------------------------------------
# 1.  Generate a random plane (two independent columns) and point b
# ------------------------------------------------------------------
def generate_data(seed=None):
    if seed is not None:
        np.random.seed(seed)
    # Two random vectors → span a plane
    while True:
        a1 = np.random.randn(3)
        a2 = np.random.randn(3)
        if np.linalg.matrix_rank(np.column_stack([a1, a2])) == 2:
            break
    A = np.column_stack([a1, a2])
    A = A / np.linalg.norm(A, axis=0)          # normalize for nice scale
    b = np.random.randn(3) + 3                 # point not in plane
    return A, b

# Change the seed below to get a new random example
A, b = generate_data(seed=42)                  # ← edit this number!

# ------------------------------------------------------------------
# 2.  Compute projection p = A (A^T A)^(-1) A^T b
# ------------------------------------------------------------------
ATA_inv = np.linalg.inv(A.T @ A)
P = A @ ATA_inv @ A.T
p = P @ b
e = b - p                                      # error vector

# ------------------------------------------------------------------
# 3.  Helper: 3D arrow for the error vector
# ------------------------------------------------------------------
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)

# ------------------------------------------------------------------
# 4.  Plot everything
# ------------------------------------------------------------------
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Projection of $b$ onto a Random Plane', fontsize=14, pad=20)

# --- Plane (semi-transparent mesh) ---
s = np.linspace(-2.5, 2.5, 20)
t = np.linspace(-2.5, 2.5, 20)
S, T = np.meshgrid(s, t)
X = S * A[0,0] + T * A[0,1]
Y = S * A[1,0] + T * A[1,1]
Z = S * A[2,0] + T * A[2,1]
ax.plot_surface(X, Y, Z, color='lightblue', alpha=0.5, linewidth=0, antialiased=True)

# --- Points ---
ax.scatter(*b, color='red', s=100, label='$b$ (original point)', depthshade=False)
ax.scatter(*p, color='blue', s=100, label='$p$ (projection)', depthshade=False)
ax.scatter(0, 0, 0, color='gray', s=30, alpha=0.6)

# --- Error vector e = b - p (dashed purple arrow) ---
arrow = Arrow3D(
    [b[0], p[0]], [b[1], p[1]], [b[2], p[2]],
    mutation_scale=20, arrowstyle='-|>', color='purple', linewidth=2.5, linestyle='--'
)
ax.add_artist(arrow)

# --- Dotted perpendicular line from b to p ---
ax.plot([b[0], p[0]], [b[1], p[1]], [b[2], p[2]], 'k:', linewidth=1.8)

# --- Basis vectors of the plane (thin gray) ---
for i in range(2):
    vec = 3.0 * A[:, i]
    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color='gray', alpha=0.7, linewidth=1.2)

# --- Labels ---
ax.text(b[0], b[1], b[2], "  $b$", color='red', fontsize=13, weight='bold')
ax.text(p[0], p[1], p[2], "  $p$", color='blue', fontsize=13, weight='bold')
mid = (b + p) / 2
ax.text(mid[0], mid[1], mid[2], "  $e$", color='purple', fontsize=13, weight='bold')

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.legend(loc='upper left', fontsize=11)
ax.set_xlim([-4, 4]); ax.set_ylim([-4, 4]); ax.set_zlim([-4, 4])
ax.view_init(elev=20, azim=30)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------
# 5.  Summary in Markdown (auto-updated when cell is re-run)
# ------------------------------------------------------------------
def latex_vec(v, name, dec=2):
    entries = [f"{x:.{dec}f}" for x in v]
    return f"${name} = \\begin{{pmatrix}} {entries[0]} \\\\ {entries[1]} \\\\ {entries[2]} \\end{{pmatrix}}$"

summary = f"""
### Projection Summary

**Plane basis** (columns of $A$):  
{latex_vec(A[:,0], 'a_1')}, {latex_vec(A[:,1], 'a_2')}

**Point** $b$:  
{latex_vec(b, 'b')}

**Projection** $p = A (A^T A)^{-1} A^T b$:  
{latex_vec(p, 'p')}

**Error** $e = b - p$:  
{latex_vec(e, 'e')}  
$\|e\| = {np.linalg.norm(e):.3f}$

**Orthogonality check** (should be ≈0):  
$a_1^T e = {A[:,0]@e:.2e}$,  
$a_2^T e = {A[:,1]@e:.2e}$

---

*To see a new random example, just **re-run this cell** (or change the `seed` value above).*
"""

display(Markdown(summary))
```

## Lecture 16: Projection Matrices

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/osh80YCg_GM"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

In this lecture we explore projection matrices and least-squares problems, focusing on projecting vectors onto subspaces and finding approximate solutions to $Ax = b$. It builds on Lecture 15’s projection concepts, and we can start to connect these concepts in linear algebra back to model fitting and optimization.

### Key Definitions and Geometric Intuitions
**Projection Matrix**: \
Definition: For $C(A)$ spanned by columns of $A$ (m x n, full column rank), the projection matrix is $P = A (A^T A)^{-1} A^T$. \
Geometric Intuition: $P b$ projects $b$ onto $C(A)$, the closest point in the subspace. P4 is symmetric $( P^T = P )$ and idempotent $( P^2 = P )$, acting like a “flattening” operation onto $C(A)$.

**Least-Squares Problem**: \
Definition: Minimize $| Ax - b |_2 $ when $ b \notin C(A)$. Solution: $\hat{x} = (A^T A)^{-1} A^T b $, with projection $ p = A \hat{x} $. \
Geometric Intuition: $p$ is the closest point in $C(A)$ to $b$, with error $e = b - p \perp C(A)$, lying in $N(A^T)$. \

**Orthogonality**: \
Definition: The error $e = b - A \hat{x}$ satisfies $A^T e = 0 $. \
Geometric Intuition: The error is perpendicular to $C(A)$, ensuring the shortest distance to $b$. \

Applications:
1. Least-squares solutions for data fitting (e.g., linear regression).
2. Optimization problems where exact solutions are not feasible.

Example from last lecture:
For $A = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{pmatrix}$, $b = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$: \
Projection matrix: $ P = A A^T = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}$. \
Projection: $ p = P b = \begin{pmatrix} 1 \\ 2 \\ 0 \end{pmatrix} $. \
Least-squares: $ \hat{x} = (A^T A)^{-1} A^T b = \begin{pmatrix} 1 \\ 2 \end{pmatrix} $, so $ A \hat{x} = \begin{pmatrix} 1 \\ 2 \\ 0 \end{pmatrix}$ .\
Error: $e = b - p = \begin{pmatrix} 0 \\ 0 \\ 3 \end{pmatrix}$, orthogonal to $C(A)$.

We can now ask the question: which choice of A in $R^(m \cross n)$ minimizes the error?

Interactive Problem Generator

This code generates a random 3x2 matrix A (rank 2) and vector b, computes the projection matrix P, projects b onto C(A), and finds the least-squares solution $\hat{x}$. It visualizes b, p, and e in 3D. Predict the projection, least-squares solution, and verify orthogonality, then check the output.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import Markdown, display

def generate_matrix_and_b():
    """
    Generate a random 3x2 matrix with rank 2 and a random b.
    """
    v1 = np.random.randint(-5, 6, 3)
    v2 = np.random.randint(-5, 6, 3)
    while np.linalg.matrix_rank(np.column_stack((v1, v2))) != 2:
        v2 = np.random.randint(-5, 6, 3)
    A = np.column_stack((v1, v2))
    b = np.random.randint(-5, 6, 3)
    return A, b

def compute_projection(A, b):
    """
    Compute projection matrix, projection of b onto C(A), and least-squares solution.
    """
    P = np.dot(A, np.dot(np.linalg.inv(np.dot(A.T, A)), A.T))
    x_hat = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
    p = np.dot(A, x_hat)
    e = b - p
    return P, x_hat, p, e

def matrix_to_latex(M):
    rows = [r" & ".join([f"{x:.2f}" if abs(x) > 1e-10 else "0" for x in row]) for row in M]
    return r"\begin{pmatrix} " + r" \\ ".join(rows) + r"\end{pmatrix}"

def plot_projection(A, b, p, e):
    """
    Plot b, p, and e in 3D.
    """
    Q = np.linalg.qr(A)[0]  # Orthonormal basis for C(A)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot origin
    ax.scatter([0], [0], [0], color='black', s=50, label='Origin')
    
    # Plot b, p, e as vectors
    ax.quiver(0, 0, 0, b[0], b[1], b[2], color='blue', label='b', linewidth=2)
    ax.quiver(0, 0, 0, p[0], p[1], p[2], color='green', label='Projection p', linewidth=2)
    ax.quiver(p[0], p[1], p[2], e[0], e[1], e[2], color='red', label='Error e', linewidth=2)
    
    # Plot column space plane
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = X[i, j] * Q[:, 0] + Y[i, j] * Q[:, 1]
            Z[i, j] = point[2]
            X[i, j] = point[0]
            Y[i, j] = point[1]
    ax.plot_surface(X, Y, Z, color='cyan', alpha=0.3, label='C(A)')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    max_range = np.max(np.abs(np.concatenate([b, p, e]))) * 1.5
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    ax.legend()
    plt.title('Least-Squares Projection of b onto C(A)')
    plt.show()

# Generate matrix and vector
A, b = generate_matrix_and_b()
P, x_hat, p, e = compute_projection(A, b)

# Verify properties
symmetric = np.allclose(P, P.T)
idempotent = np.allclose(np.dot(P, P), P)
ortho_check = np.allclose(np.dot(A.T, e), 0)

# Display results
markdown = f"**Matrix A**:\n$$\n{matrix_to_latex(A)}\n$$\n\n"
markdown += f"**Vector b**:\n$$\n{matrix_to_latex(b.reshape(-1, 1))}\n$$\n\n"
display(Markdown(markdown))
plot_projection(A, b, p, e)
markdown = f"Predict:\n- Least-squares solution x̂.\n- Projection p = A x̂.\n- Error e = b - p.\n- Is P symmetric and idempotent?\n- Is e ⊥ C(A)?\n\n"
display(Markdown(markdown))

# Reveal answers (students uncomment after predicting)
# markdown = f"**Answers**:\n"
# markdown += f"- Least-squares solution x̂:\n$$\n{matrix_to_latex(x_hat.reshape(-1, 1))}\n$$\n"
# markdown += f"- Projection p:\n$$\n{matrix_to_latex(p.reshape(-1, 1))}\n$$\n"
# markdown += f"- Error e:\n$$\n{matrix_to_latex(e.reshape(-1, 1))}\n$$\n"
# markdown += f"- P is symmetric: {'True' if symmetric else 'False'}.\n"
# markdown += f"- P is idempotent: {'True' if idempotent else 'False'}.\n"
# markdown += f"- e ⊥ C(A): {'True' if ortho_check else 'False'}.\n\n"
# display(Markdown(markdown))
```