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

### Key Definitions and Geometric Intuitions
**Orthogonal Vectors**:
Definition: Vectors $u, v \in \mathbb{R}^n$ are orthogonal if $u^T v = 0$.
Geometric Intuition: Orthogonal vectors are perpendicular (90° angle), like axes in a coordinate system. This simplifies projections, as components along orthogonal directions are independent.
**Orthogonal Subspaces**:
Definition: Subspaces $S, T \subset \mathbb{R}^n$ are orthogonal if $u^T v = 0$ for all $u \in S$, $v \in T$.
Geometric Intuition: Subspaces that are perpendicular, decomposing $\mathbb{R}^n$ into complementary parts. For a matrix $A$ (m x n), $C(A) \perp N(A^T)$ in $\mathbb{R}^m$, $C(A^T) \perp N(A)$ in $\mathbb{R}^n$, splitting the space into orthogonal components (e.g., potential differences vs. equal potentials in graphs).
**Orthogonal Basis**:

Definition: A basis for a subspace where vectors are pairwise orthogonal: $u_i^T u_j = 0$ for $i \neq j$. If normalized ($|u_i| = 1$), it’s orthonormal.

Geometric Intuition: An orthogonal basis forms a “perpendicular coordinate system” for the subspace, making vector expansions straightforward (e.g., coordinates are dot products).

**Orthogonal Complement**:

Definition: For a subspace $S \subset \mathbb{R}^n$, $S^\perp = { v \in \mathbb{R}^n \mid v^T u = 0 \text{ for all } u \in S }$. Dimension: $\dim(S) + \dim(S^\perp) = n$.

Geometric Intuition: $S^\perp$ is the subspace of all vectors perpendicular to $S$, forming a complementary space (e.g., $N(A^T) = C(A)^\perp$).

Example

For $A = \begin{pmatrix} 1 & 2 \ 2 & 4 \ 0 & 1 \end{pmatrix}$ (rank 2):

Column Space $C(A)$: Dim = 2. Basis: $\left{ \begin{pmatrix} 1 \ 2 \ 0 \end{pmatrix}, \begin{pmatrix} 2 \ 4 \ 1 \end{pmatrix} \right}$.

Orthogonal Basis: Apply Gram-Schmidt:

$u_1 = \begin{pmatrix} 1 \ 2 \ 0 \end{pmatrix}$, normalize: $q_1 = \frac{u_1}{|u_1|} = \frac{1}{\sqrt{5}} \begin{pmatrix} 1 \ 2 \ 0 \end{pmatrix}$.

$u_2 = \begin{pmatrix} 2 \ 4 \ 1 \end{pmatrix} - \text{proj}_{u_1} \begin{pmatrix} 2 \ 4 \ 1 \end{pmatrix}$, then normalize.

Left Nullspace $N(A^T)$: Dim = 3 - 2 = 1. Basis: $\begin{pmatrix} 2 \ -1 \ 0 \end{pmatrix}$. Verify: $u_1^T \begin{pmatrix} 2 \ -1 \ 0 \end{pmatrix} = 0$, confirming $C(A) \perp N(A^T)$.

```{code-cell} python
import numpy as np
from IPython.display import Markdown, display

def generate_matrix():
    """
    Generate a random 3x3 matrix with rank 2.
    """
    v1 = np.random.randint(-5, 6, 3)
    v2 = np.random.randint(-5, 6, 3)
    while np.linalg.matrix_rank(np.column_stack((v1, v2))) != 2:
        v2 = np.random.randint(-5, 6, 3)
    coeffs = np.random.randint(-2, 3, 2)
    v3 = coeffs[0] * v1 + coeffs[1] * v2
    A = np.column_stack((v1, v2, v3))
    return A

def gram_schmidt(V):
    """
    Apply Gram-Schmidt to columns of V to get an orthogonal basis.
    """
    V = V.astype(float)
    n, k = V.shape
    U = np.zeros((n, k))
    U[:, 0] = V[:, 0]
    for i in range(1, k):
        u = V[:, i]
        for j in range(i):
            u = u - (np.dot(V[:, i], U[:, j]) / np.dot(U[:, j], U[:, j])) * U[:, j]
        U[:, i] = u
    # Normalize to orthonormal
    Q = np.zeros((n, k))
    for i in range(k):
        if np.linalg.norm(U[:, i]) > 1e-10:
            Q[:, i] = U[:, i] / np.linalg.norm(U[:, i])
    return Q

def matrix_to_latex(M):
    rows = [r" & ".join([f"{x:.2f}" if abs(x) > 1e-10 else "0" for x in row]) for row in M]
    return r"\begin{pmatrix} " + r" \\ ".join(rows) + r"\end{pmatrix}"

# Generate matrix and compute subspaces
A = generate_matrix()
R, pivot_cols = np.linalg.matrix_rank(A), [0, 1]  # Assume first two columns are independent
c_a_basis = A[:, pivot_cols]
U, S, Vt = np.linalg.svd(A.T, full_matrices=True)
n_at_basis = Vt[rank:, :].T if rank < A.shape[0] else np.zeros((A.shape[0], 1))

# Apply Gram-Schmidt to C(A) basis
ortho_basis = gram_schmidt(c_a_basis)

# Verify orthogonality
ortho_check = np.allclose(np.dot(ortho_basis[:, 0], ortho_basis[:, 1]), 0)
c_a_n_at_ortho = all(np.allclose(np.dot(c_a_basis[:, i], n_at_basis[:, j]), 0) for i in range(c_a_basis.shape[1]) for j in range(n_at_basis.shape[1]) if n_at_basis.shape[1] > 0)

# Display results
markdown = f"**Matrix A**:\n$\n{matrix_to_latex(A)}\n$\n\n"
markdown += f"**Column Space Basis**:\n$\n{matrix_to_latex(c_a_basis)}\n$\n\n"
markdown += f"Predict:\n- Orthogonal basis for C(A) (apply Gram-Schmidt).\n- Is C(A) orthogonal to N(A^T)?\n- Dimensions: dim(C(A)), dim(N(A^T)).\n\n"
display(Markdown(markdown))

# Reveal answers (students uncomment after predicting)
# markdown = f"**Answers**:\n"
# markdown += f"- Orthogonal Basis for C(A):\n$\n{matrix_to_latex(ortho_basis)}\n$\n"
# markdown += f"- Orthogonality Check: Columns are {'orthogonal' if ortho_check else 'not orthogonal'}.\n"
# markdown += f"- C(A) ⊥ N(A^T): {'True' if c_a_n_at_ortho else 'False'}.\n"
# markdown += f"- dim(C(A)) = {rank}, dim(N(A^T)) = {A.shape[0] - rank}.\n\n"
# display(Markdown(markdown))
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
