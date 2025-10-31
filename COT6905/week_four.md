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
$$
u^{T}v = 0 .
$$
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
* Write the equation row-by-row:  
  $$
  \text{(row}_i \text{ of }A) \cdot x = 0 \quad \text{for every } i.
  $$
* Every row of $A$ (i.e. every vector in the row space) is orthogonal to $x$.  
* Because this holds for **all** $x\in N(A)$, the whole row space is orthogonal to the whole null space.

The same argument (applied to $A^{T}$) shows $C(A) \perp N(A^{T})$.

### 4. Orthogonal Complements  
For a subspace $S\subset \mathbb{R}^k$ its **orthogonal complement** is  
$$
S^{\perp}= \{ v\in\mathbb{R}^k \mid v^{T}u=0 \;\forall u\in S\}.
$$
Key fact:  
$$
\dim S + \dim S^{\perp} = k.
$$

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

This lecture introduces projections onto subspaces, finding the closest point in a subspace to a vector. We explore the projection matrix and orthogonal bases, building on Lecture 14’s orthogonality concepts.

### Key Definitions and Geometric Intuitions

**Projection onto a Subspace**:

Definition: For $b \in \mathbb{R}^n$ and subspace $S \subset \mathbb{R}^n$, the projection $p \in S$ is the closest point to $b$, with error $e = b - p \perp S$.

Geometric Intuition: $p$ is the “shadow” of $b$ onto $S$ along perpendicular lines. The error $e$ lies in $S^\perp$, ensuring the shortest distance.

**Projection Matrix**:

Definition: For a subspace $C(A)$ (where $A$ is m x n, full column rank), the projection matrix is $P = A (A^T A)^{-1} A^T$.

Geometric Intuition: $P b$ projects $b$ onto $C(A)$, with $P$ symmetric ($P^T = P$) and idempotent ($P^2 = P$). It maps vectors to their closest points in $C(A)$.

**Orthogonal Projection**:

Definition: Using an orthonormal basis $q_1, \ldots, q_k$ for $S$, $p = \sum_{i=1}^k (q_i^T b) q_i$.

Geometric Intuition: Each term $(q_i^T b) q_i$ is the component of $b$ along $q_i$, summing to the projection.

**Orthogonal Complement**:

Definition: $S^\perp = { v \in \mathbb{R}^n \mid v^T u = 0 \text{ for all } u \in S }$, with $\dim(S) + \dim(S^\perp) = n$.

Geometric Intuition: $S^\perp$ contains all vectors perpendicular to $S$, e.g., $N(A^T)$ for $C(A)$.

Example

For $A = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{pmatrix}$, project $b = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$ onto $C(A)$:

**Orthonormal basis**: $\begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}$, $\begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}$.

Projection: $p = (1 \cdot 1) \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix} + (2 \cdot 1) \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix} = \begin{pmatrix} 1 \\ 2 \\ 0 \end{pmatrix}$.

Error: $e = b - p = \begin{pmatrix} 0 \\ 0 \\ 3 \end{pmatrix}$, orthogonal to $C(A)$.

Interactive Problem Generator

This code generates a random 3x2 matrix $A$ (rank 2), computes an orthonormal basis for $C(A)$, projects a random $b$ onto $C(A)$, and visualizes $b$, $p$, and $e$ in 3D. Predict the projection and verify orthogonality, then check the output.

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

def gram_schmidt(A):
    """
    Apply Gram-Schmidt to columns of A to get an orthonormal basis.
    """
    A = A.astype(float)
    n, k = A.shape
    U = np.zeros((n, k))
    U[:, 0] = A[:, 0]
    for i in range(1, k):
        u = A[:, i]
        for j in range(i):
            u = u - (np.dot(A[:, i], U[:, j]) / np.dot(U[:, j], U[:, j])) * U[:, j]
        U[:, i] = u
    Q = np.zeros((n, k))
    for i in range(k):
        if np.linalg.norm(U[:, i]) > 1e-10:
            Q[:, i] = U[:, i] / np.linalg.norm(U[:, i])
    return Q

def project_onto_subspace(A, b):
    """
    Project b onto C(A) and compute the projection matrix.
    """
    Q = gram_schmidt(A)
    p = np.dot(Q, np.dot(Q.T, b))
    P = np.dot(Q, Q.T)
    e = b - p
    return p, e, P

def matrix_to_latex(M):
    rows = [r" & ".join([f"{x:.2f}" if abs(x) > 1e-10 else "0" for x in row]) for row in M]
    return r"\begin{pmatrix} " + r" \\ ".join(rows) + r"\end{pmatrix}"

def plot_projection(A, b, p, e):
    """
    Plot b, p, and e in 3D.
    """
    Q = gram_schmidt(A)
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
    plt.title('Projection of b onto C(A)')
    plt.show()

# Generate matrix and vector
A, b = generate_matrix_and_b()
p, e, P = project_onto_subspace(A, b)

# Verify orthogonality
Q = gram_schmidt(A)
ortho_check = np.allclose(np.dot(Q.T, e), 0)

# Display results
markdown = f"**Matrix A**:\n$$\n{matrix_to_latex(A)}\n$$\n\n"
markdown += f"**Vector b**:\n$$\n{matrix_to_latex(b.reshape(-1, 1))}\n$$\n\n"
display(Markdown(markdown))
plot_projection(A, b, p, e)
markdown = f"Predict:\n- Projection p of b onto C(A).\n- Error e = b - p.\n- Is e orthogonal to C(A)?\n- Dimension of C(A)?\n\n"
display(Markdown(markdown))

# Reveal answers
# markdown = f"**Answers**:\n"
# markdown += f"- Projection p:\n$$\n{matrix_to_latex(p.reshape(-1, 1))}\n$$\n"
# markdown += f"- Error e:\n$$\n{matrix_to_latex(e.reshape(-1, 1))}\n$$\n"
# markdown += f"- e ⊥ C(A): {'True' if ortho_check else 'False'}.\n"
# markdown += f"- dim(C(A)) = {np.linalg.matrix_rank(A)}.\n\n"
# display(Markdown(markdown))
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