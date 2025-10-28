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

# Week 3: Independence, Spaces, Graphs

## Lecture 9: Independence, basis, and dimension

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/yjBerM5jWsc"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

This lecture introduces linear independence, basis, and dimension, key concepts for understanding vector spaces and subspaces like $C(A)$ and $N(A)$. These ideas help us describe the structure of matrices and their solutions, building on Lectures 6–8.

### Key Definitions
**Linear Independence**: A set of vectors ${v_1, v_2, \ldots, v_k}$ is independent if $c_1 v_1 + c_2 v_2 + \cdots + c_k v_k = 0$ only when $c_1 = c_2 = \cdots = c_k = 0$ Otherwise, they are dependent.

**Basis**: A set of linearly independent vectors that spans a vector space (every vector in the space is a unique linear combination of the basis).

**Dimension**: The number of vectors in a basis. For a matrix $A$ (m x n): \
$\dim(C(A)) = \text{rank}(A)$ \
$\dim(N(A)) = n - \text{rank}(A)$ \
We call this relationship the **Rank-Nullity Theorem**: $\text{rank}(A) + \dim(N(A)) = n $

**Example**

For $A = \begin{pmatrix} 1 & 2 & 2 \\ 3 & 4 & 6 \\ 5 & 6 & 10 \end{pmatrix} $

RREF: $\begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}$\
Rank: 2 (two pivot columns). \
Columns 1 and 2 are independent (pivot columns). Column 3 is dependent (e.g., $2 v_1 + v_2 - v_3 = 0 $) \
Basis for $C(A)$: $\begin{pmatrix} 1 \\ 3 \\ 5 \end{pmatrix}, \begin{pmatrix} 2 \\ 4 \\ 6 \end{pmatrix} $ \
Dimensions: $\dim(C(A)) = 2 $, $ \dim(N(A)) = 3 - 2 = 1 $

Interactive Problem Generator

This code generates two 3x3 matrices:

Rank 2: Two independent columns.

Rank 1: All columns dependent (multiples of one vector). For each, compute the RREF and predict:

Are the columns linearly independent?

What is a basis for $C(A)$?

What are $\dim(C(A))$ and $\dim(N(A))$? Check the output to verify.

```{code-cell} python
import numpy as np
from sympy import Matrix
from IPython.display import Markdown, display

def generate_matrix(case='rank2'):
    """
    Generate a random 3x3 matrix with specified rank.
    """
    if case == 'rank2':
        # Rank 2: Two independent columns
        v1 = np.random.randint(-5, 6, 3)
        v2 = np.random.randint(-5, 6, 3)
        while np.linalg.matrix_rank(np.column_stack((v1, v2))) != 2:
            v2 = np.random.randint(-5, 6, 3)
        coeffs = np.random.randint(-2, 3, 2)
        v3 = coeffs[0] * v1 + coeffs[1] * v2
        A = np.column_stack((v1, v2, v3))
    elif case == 'rank1':
        # Rank 1: All columns dependent
        v1 = np.random.randint(-5, 6, 3)
        while np.all(v1 == 0):
            v1 = np.random.randint(-5, 6, 3)
        scalars = np.random.randint(-2, 3, 2)
        v2 = scalars[0] * v1
        v3 = scalars[1] * v1
        A = np.column_stack((v1, v2, v3))
    else:
        raise ValueError("Case must be 'rank2' or 'rank1'")
    return A

def matrix_to_latex(M):
    """
    Convert matrix to LaTeX, handling SymPy rationals or NumPy floats.
    """
    if isinstance(M, Matrix):
        rows = [r" & ".join([str(x) for x in row]) for row in M.tolist()]
    else:
        rows = [r" & ".join([f"{x:.0f}" if abs(x) > 1e-10 else "0" for x in row]) for row in M]
    return r"\begin{pmatrix} " + r" \\ ".join(rows) + r" \end{pmatrix}"

# Generate and analyze matrices
A_rank2 = generate_matrix('rank2')
A_rank2_sym = Matrix(A_rank2)
R_rank2, pivot_cols_rank2 = A_rank2_sym.rref()
R_rank2_np = np.array(R_rank2).astype(float)

A_rank1 = generate_matrix('rank1')
A_rank1_sym = Matrix(A_rank1)
R_rank1, pivot_cols_rank1 = A_rank1_sym.rref()
R_rank1_np = np.array(R_rank1).astype(float)

# Display Problem 1: Rank 2
markdown = f"**Problem 1: Rank 2 Matrix**\n\n"
markdown += f"**Matrix A**:\n$ {matrix_to_latex(A_rank2)} $\n\n"
markdown += f"**RREF of A**:\n$ {matrix_to_latex(R_rank2)} $\n\n"
markdown += f"**Pivot Columns**: {[i+1 for i in pivot_cols_rank2]} (1-based indexing)\n\n"
markdown += f"Predict:\n- Are the columns linearly independent?\n- Basis for C(A)?\n- Dimensions: dim(C(A)), dim(N(A))?\n\n"
display(Markdown(markdown))

# Display Problem 2: Rank 1
markdown = f"**Problem 2: Rank 1 Matrix**\n\n"
markdown += f"**Matrix A**:\n$ {matrix_to_latex(A_rank1)} $\n\n"
markdown += f"**RREF of A**:\n$ {matrix_to_latex(R_rank1)} $\n\n"
markdown += f"**Pivot Columns**: {[i+1 for i in pivot_cols_rank1]} (1-based indexing)\n\n"
markdown += f"Predict:\n- Are the columns linearly independent?\n- Basis for C(A)?\n- Dimensions: dim(C(A)), dim(N(A))?\n\n"
display(Markdown(markdown))

# Reveal answers (students uncomment after predicting)
# rank = A_rank2_sym.rank()
# markdown = f"**Answers for Problem 1**:\n"
# markdown += f"- Linear Independence: Columns {[i+1 for i in pivot_cols_rank2]} are independent; others dependent.\n"
# markdown += f"- Basis for C(A): Columns {[i+1 for i in pivot_cols_rank2]} of A.\n"
# markdown += f"- dim(C(A)) = {rank}, dim(N(A)) = {A_rank2.shape[1] - rank}.\n\n"
# display(Markdown(markdown))
#
# rank = A_rank1_sym.rank()
# markdown = f"**Answers for Problem 2**:\n"
# markdown += f"- Linear Independence: Columns are dependent (rank 1).\n"
# markdown += f"- Basis for C(A): Column {[pivot_cols_rank1[0]+1 if pivot_cols_rank1 else 'none']} of A.\n"
# markdown += f"- dim(C(A)) = {rank}, dim(N(A)) = {A_rank1.shape[1] - rank}.\n\n"
# display(Markdown(markdown))
```

## Lecture 10: The four fundamental subspaces

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/nHlE7EgJFds"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

This lecture introduces the four fundamental subspaces of an $m \times n$ matrix $A$: column space $C(A)$, nullspace $N(A)$, row space $C(A^T)$, and left nullspace $N(A^T)$. We explore their dimensions, bases, and orthogonality, connecting to the rank-nullity theorem and linear system solutions.

### Key Definitions

**Column Space** $C(A)$: Subspace of $\mathbb{R}^m$, spanned by $A$’s columns. Dimension: $\text{rank}(A)$ \
**Nullspace** $N(A)$: Subspace of $\mathbb{R}^n$, all $x$ such that $Ax = 0$. Dimension: $n - \text{rank}(A)$ \
**Row Space** $C(A^T)$: Subspace of $\mathbb{R}^n$, spanned by $A$’s rows columns of $A^T$. Dimension: $\text{rank}(A)$ \
**Left Nullspace** $N(A^T)$: Subspace of $\mathbb{R}^m$, all $y$ such that $A^T y = 0$. Dimension: $m - \text{rank}(A)$ \
**Orthogonality**: $C(A) \perp N(A^T)$ in $ \mathbb{R}^m $, $C(A^T) \perp N(A)$ in $\mathbb{R}^n$ \
**Rank-Nullity**: $\text{rank}(A) + \dim(N(A)) = n $, $\text{rank}(A^T) + \dim(N(A^T)) = m$

### Example

For $A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 3 & 6 & 9 \end{pmatrix} $ (rank 1): \
RREF: $\begin{pmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix} $ \
Column Space: $C(A) = \text{span}{ \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix} }$, dim = 1 \
Nullspace: Basis $\begin{pmatrix} -2 \\ 1 \\ 0 \end{pmatrix}$, $\begin{pmatrix} -3 \\ 0 \\ 1 \end{pmatrix}$, $dim = 3 - 1 = 2 $ \
Row Space: $C(A^T) = \text{span}{ \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix} }$, $dim = 1$ \
Left Nullspace: Basis $\begin{pmatrix} -2 \\ 1 \\ 0 \end{pmatrix}$, $\begin{pmatrix} -3 \\ 0 \\ 1 \end{pmatrix}$, $dim = 2 - 1 = 1$


```{code-cell} python
import numpy as np
from sympy import Matrix
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

def find_subspaces(A):
    A_sym = Matrix(A)
    A_t_sym = A_sym.T
    
    # Column space and nullspace
    R, pivot_cols = A_sym.rref()
    rank = len(pivot_cols)
    c_a_basis = [np.array(A_sym.col(j)).flatten().astype(float) for j in pivot_cols]
    n_a_basis = [np.array(vec).flatten().astype(float) for vec in A_sym.nullspace()]
    
    # Row space and left nullspace
    R_at, pivot_cols_at = A_t_sym.rref()
    c_at_basis = [np.array(A_t_sym.col(j)).flatten().astype(float) for j in pivot_cols_at]
    n_at_basis = [np.array(vec).flatten().astype(float) for vec in A_t_sym.nullspace()]
    
    return rank, c_a_basis, n_a_basis, c_at_basis, n_at_basis

# 3x3 rank-1 matrix
A = np.array([[1, 2, 3],
              [2, 4, 6],
              [3, 6, 9]])
rank, c_a_basis, n_a_basis, c_at_basis, n_at_basis = find_subspaces(A)

# Display summary
markdown = f"**Matrix A (Rank {rank}):**\n\n"
markdown += r"$ A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 3 & 6 & 9 \end{pmatrix} $\n\n"
markdown += f"- dim(C(A)) = {rank}, Basis: `{c_a_basis[0]}`\n"
markdown += f"- dim(N(A)) = {3 - rank}, Basis: `{n_a_basis[0]}`, `{n_a_basis[1]}`\n"
markdown += f"- dim(C(Aᵀ)) = {rank}, Basis: `{c_at_basis[0]}`\n"
markdown += f"- dim(N(Aᵀ)) = {3 - rank}, Basis: `{n_at_basis[0]}`, `{n_at_basis[1]}`\n"
display(Markdown(markdown))

# === 3D Visualization (Modern Style - No mpl_toolkits import) ===
fig = plt.figure(figsize=(14, 6))

# Left: Domain R^3 — Row Space (line), Null Space (plane)
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title('Domain $\\mathbb{R}^3$: Row Space (blue), Null Space (green)')

# Row space: line through origin
row_vec = c_at_basis[0]
t = np.linspace(-6, 6, 100)
ax1.plot(t * row_vec[0], t * row_vec[1], t * row_vec[2], 'b-', linewidth=3, label='Row Space')

# Null space: plane
b1, b2 = n_a_basis
s, t_grid = np.meshgrid(np.linspace(-3, 3, 8), np.linspace(-3, 3, 8))
X = s * b1[0] + t_grid * b2[0]
Y = s * b1[1] + t_grid * b2[1]
Z = s * b1[2] + t_grid * b2[2]
ax1.plot_surface(X, Y, Z, color='green', alpha=0.4)

ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('z')
ax1.legend()
ax1.set_xlim([-10,10]); ax1.set_ylim([-10,10]); ax1.set_zlim([-10,10])

# Right: Codomain R^3 — Column Space (line), Left Null Space (plane)
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title('Codomain $\\mathbb{R}^3$: Column Space (red), Left Null Space (purple)')

# Column space: line
col_vec = c_a_basis[0]
ax2.plot(t * col_vec[0], t * col_vec[1], t * col_vec[2], 'r-', linewidth=3, label='Column Space')

# Left null space: plane
b1_l, b2_l = n_at_basis
X_l = s * b1_l[0] + t_grid * b2_l[0]
Y_l = s * b1_l[1] + t_grid * b2_l[1]
Z_l = s * b1_l[2] + t_grid * b2_l[2]
ax2.plot_surface(X_l, Y_l, Z_l, color='purple', alpha=0.4)

ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z')
ax2.legend()
ax2.set_xlim([-10,10]); ax2.set_ylim([-10,10]); ax2.set_zlim([-10,10])

plt.tight_layout()
plt.show()
```

## Lecture 11: Matrix spaces; rank 1; small world graphs

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/2IdtqGM6KWU"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

This lecture explores matrix spaces as vector spaces, the structure of rank-1 matrices, and small world graphs via adjacency matrices. These concepts extend our understanding of subspaces and apply linear algebra to networks.

## Lecture 11: Matrix spaces; rank 1; small world graphs
<iframe width="560" height="315"
    src="https://www.youtube.com/embed/2IdtqGM6KWU"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

This lecture expands the notion of **vector spaces** to collections of matrices, proves which collections are subspaces, and connects rank-1 matrices to low-rank structure. It closes with an introduction to **small-world graphs** via adjacency matrices, illustrating how linear-algebraic tools describe network connectivity.

### Key Definitions

**Matrix Space** $M_{m \times n}$:  
The set of *all* $m \times n$ real matrices, equipped with componentwise addition and scalar multiplication. It is a vector space of dimension $m \cdot n$.

**Symmetric Matrices** $\mathcal{S}_{n}$:  
Matrices $A \in M_{n \times n}$ satisfying $A = A^T$.  

**Upper-Triangular Matrices** $\mathcal{U}_{n}$:  
Matrices $A \in M_{n \times n}$ with $A_{ij}=0$ whenever $i>j$.

**Diagonal Matrices** $\mathcal{D}_{n}$:  
Matrices $A \in M_{n \times n}$ with $A_{ij}=0$ whenever $i\neq j$.

**Rank-1 Matrix**:  
$A = u v^T$ for column vectors $u \in \mathbb{R}^m$, $v \in \mathbb{R}^n$ (both non-zero).  
All rows of $A$ are scalar multiples of $v^T$; all columns are scalar multiples of $u$.

**Adjacency Matrix** of a graph:  
For a graph with $n$ nodes, the $n \times n$ matrix $A$ where  
\[
A_{ij}=
\begin{cases}
1 & \text{if nodes $i$ and $j$ are connected},\\
0 & \text{otherwise}.
\end{cases}
\]
The entry $(A^k)_{ij}$ counts walks of length $k$ from node $i$ to node $j$.

**Small-World Graph**:  
A graph in which the typical shortest path between nodes is short (often logarithmic in the number of nodes), e.g., the "six degrees of separation" phenomenon.

### Subspaces of $M_{n \times n}$

| Set | Subspace? | Reason | Dimension (for $n=3$) |
|-----|-----------|--------|-----------------------|
| Symmetric $\mathcal{S}_3$ | Yes | Closed under addition & scalar multiplication | $6$ |
| Upper-triangular $\mathcal{U}_3$ | Yes | Same closure properties | $6$ |
| Diagonal $\mathcal{D}_3$ | Yes | Intersection $\mathcal{S}_3 \cap \mathcal{U}_3$ | $3$ |
| Union $\mathcal{S}_3 \cup \mathcal{U}_3$ | No | Not closed under addition | — |
| Span $\mathcal{S}_3 + \mathcal{U}_3$ | Yes | Spans all of $M_{3 \times 3}$ | $9$ |

**Proof sketch for symmetric matrices**  
If $A = A^T$ and $B = B^T$, then $(A+B)^T = A^T + B^T = A + B$ and $(\lambda A)^T = \lambda A^T = \lambda A$. Hence $\mathcal{S}_n$ is a subspace.

**Proof sketch for upper-triangular**  
If $A_{ij}=0$ for $i>j$ and same for $B$, then $(A+B)_{ij}=0$ for $i>j$ and $(\lambda A)_{ij}=0$ for $i>j$.

**Intersection = diagonal**  
A matrix that is both symmetric and upper-triangular must have zeros below *and* above the diagonal, leaving only the diagonal entries free.

**Union is not a subspace**  
Counter-example:  
\[
\begin{pmatrix} 1 & 1 \\ 1 & 0 \end{pmatrix} \in \mathcal{S}_2,
\qquad
\begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix} \in \mathcal{U}_2,
\]
but their sum
\[
\begin{pmatrix} 2 & 2 \\ 1 & 0 \end{pmatrix}
\]
is neither symmetric nor upper-triangular.

**Span of symmetric + upper-triangular = all matrices**  
Any matrix $M$ can be decomposed as  
\[
M = \frac{M+M^T}{2} + \left(M - \frac{M+M^T}{2}\right),
\]
where the first term is symmetric and the second is upper-triangular (its lower part cancels). Thus the span is $M_{n \times n}$.

### Rank-1 Matrices

A matrix $A \in M_{m \times n}$ has rank 1 **iff** it can be written $A = u v^T$ with $u \neq 0$, $v \neq 0$.  
*Column space* = $\operatorname{span}\{u\}$, dimension 1.  
*Row space* = $\operatorname{span}\{v\}$, dimension 1.

**Example**  
\[
u = \begin{pmatrix} 1 \\ 2 \end{pmatrix},\;
v = \begin{pmatrix} 3 \\ 4 \end{pmatrix}
\;\Rightarrow\;
A = u v^T = \begin{pmatrix} 3 & 4 \\ 6 & 8 \end{pmatrix}.
\]

**Sum of rank-1 matrices**  
Every matrix of rank $r$ is a sum of (at most) $r$ rank-1 matrices (outer-product form of the SVD).

**Counter-example for rank addition**  
In $M_{5 \times 17}$, let $A$ and $B$ each have rank 4. Their sum can have rank as low as 0 (if $B = -A$) or up to 8, but **not guaranteed** to be 4.

### Subspace of Vectors Summing to Zero

Consider $\mathbb{R}^4$ and the hyperplane  
\[
S = \{v = (v_1,v_2,v_3,v_4) \mid v_1+v_2+v_3+v_4 = 0\}.
\]
This is the **null space** of the row vector $a^T = [1\;1\;1\;1]$, i.e., $S = N(A)$ where  
\[
A = \begin{pmatrix} 1 & 1 & 1 & 1 \end{pmatrix}.
\]
Rank of $A$ is 1, so $\dim S = 4 - 1 = 3$.  

A basis is  
\[
\{(-1,1,0,0),\; (-1,0,1,0),\; (-1,0,0,1)\}.
\]

### Differential Equations as Vector Spaces

The general solution to the linear ODE  
\[
\frac{d^2 y}{dx^2} + y = 0
\]
forms a 2-dimensional vector space. A basis is  
\[
\{\cos x,\; \sin x\}.
\]
Any solution is $c_1 \cos x + c_2 \sin x$.

### Small-World Graphs

A **graph** $G = (V,E)$ consists of nodes $V$ and edges $E$.  
The **adjacency matrix** $A \in M_{n \times n}$ (binary, undirected, no self-loops) encodes connectivity.  
Powers of $A$ count walks:  
\[
(A^k)_{ij} = \text{number of walks of length $k$ from node $i$ to node $j$}.
\]

**Small-world property**:  
Most pairs of nodes are connected by a short path (path length $\sim \log n$). Empirically observed as "six degrees of separation" in social networks.

**Linear-algebraic view**  
The rank of $A$ reveals the number of connected components (for undirected graphs, rank = number of components if the graph is bipartite; otherwise related to the number of independent cycles). Low-rank approximations of $A$ capture dominant community structure.

Interactive Problem Generator

The following code generates: A 3x3 rank-1 matrix, verifying its rank and structure. Also, a 3x3 adjacency matrix for a simple small-world graph, computing 2-step paths. Predict the rank, basis for the matrix space, and number of 2-step paths between nodes, then check the output.

```{code-cell} python
import numpy as np
from IPython.display import Markdown, display

def generate_rank1_matrix(m=3, n=3):
    """
    Generate a random 3x3 rank-1 matrix as u v^T.
    """
    u = np.random.randint(-5, 6, m)
    while np.all(u == 0):
        u = np.random.randint(-5, 6, m)
    v = np.random.randint(-5, 6, n)
    while np.all(v == 0):
        v = np.random.randint(-5, 6, n)
    A = np.outer(u, v)
    return A, u, v

def generate_adjacency_matrix(n=3):
    """
    Generate a random 3x3 adjacency matrix for a simple graph (no self-loops).
    """
    A = np.random.randint(0, 2, (n, n))
    A = np.triu(A, 1) + np.triu(A, 1).T  # Symmetric, no diagonal
    return A

def matrix_to_latex(M):
    rows = [r" & ".join([f"{x:.0f}" if abs(x) > 1e-10 else "0" for x in row]) for row in M]
    return r"\begin{pmatrix} " + r" \\ ".join(rows) + r"\end{pmatrix}"

# Problem 1: Rank-1 Matrix
A_rank1, u, v = generate_rank1_matrix()
rank = np.linalg.matrix_rank(A_rank1)

markdown = f"**Problem 1: Rank-1 Matrix**\n\n"
markdown += f"**Matrix A**:\n$ {matrix_to_latex(A_rank1)} $\n\n"
markdown += f"Predict:\n- What is the rank of A?\n- Express A as u v^T (find u, v).\n- What is the dimension of the 3x3 matrix space M_{{3x3}}?\n\n"

display(Markdown(markdown))

# Problem 2: Adjacency Matrix
A_adj = generate_adjacency_matrix()
A_adj2 = np.dot(A_adj, A_adj)

markdown = f"**Problem 2: Adjacency Matrix for a Graph**\n\n"
markdown += f"**Adjacency Matrix A**:\n$ {matrix_to_latex(A_adj)} $\n\n"
markdown += f"Predict:\n- How many 2-step paths exist between each pair of nodes (compute A^2)?\n- What does A^2’s diagonal represent?\n\n"

display(Markdown(markdown))

# Reveal answers (students uncomment after predicting)
# markdown = f"**Answers for Problem 1**:\n"
# markdown += f"- Rank of A: {rank}\n"
# markdown += f"- A = u v^T, where u =\n$ {matrix_to_latex(u.reshape(-1, 1))} $\n"
# markdown += f"  v^T =\n$ {matrix_to_latex(v.reshape(1, -1))} $\n"
# markdown += f"- Dimension of M_{{3x3}}: 9\n\n"
# display(Markdown(markdown))
#
# markdown = f"**Answers for Problem 2**:\n"
# markdown += f"- A^2 (2-step paths):\n$ {matrix_to_latex(A_adj2)} $\n"
# markdown += f"- Diagonal of A^2: Number of 2-step paths from each node to itself.\n\n"
# display(Markdown(markdown))
```


## Lecture 12: Graphs, networks, incidence matrices

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/6-wh6yvk6uc"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

This lecture explores how linear algebra models graphs and networks using incidence matrices. We analyze the matrix’s subspaces to understand network properties like connectivity and flow, building on Lecture 10’s fundamental subspaces.

### Key Definitions

**Directed Graph**:  
A set of **nodes** $V = \{1,\dots,n\}$ and **directed edges** $E = \{e_1,\dots,e_m\}$, where each edge $e_k$ goes from a **tail** node $t_k$ to a **head** node $h_k$.

**Incidence Matrix** $A \in \mathbb{R}^{n \times m}$:  
- **Rows** ↔ nodes, **columns** ↔ edges.  
- For edge $e_k: t_k \to h_k$,  
  \[
  A_{ik} =
  \begin{cases}
  -1 & \text{if node } i = t_k \text{ (outgoing)}, \\
  +1 & \text{if node } i = h_k \text{ (incoming)}, \\
  0  & \text{otherwise}.
  \end{cases}
  \]

**Column Space** $C(A) \subseteq \mathbb{R}^n$:  
Spanned by the columns of $A$. Each column is a **unit flow** along one edge.  
**Interpretation**: All possible **node potential differences** (voltages) inducible by edge flows.

**Nullspace** $N(A) \subseteq \mathbb{R}^m$:  
Solutions $x \in \mathbb{R}^m$ such that $Ax = 0$.  
**Interpretation**: **Cycle flows** — edge weights that produce zero net flow at every node (conservation).

**Row Space** $C(A^T) \subseteq \mathbb{R}^m$:  
Spanned by rows of $A$ (columns of $A^T$).  
**Interpretation**: Linear combinations of node incidence patterns — valid **edge current distributions**.

**Left Nullspace** $N(A^T) \subseteq \mathbb{R}^n$:  
Solutions $y \in \mathbb{R}^n$ such that $A^T y = 0$.  
**Interpretation**: **Constant node potentials** — all nodes at the same voltage (ground).

**Kirchhoff’s Current Law (KCL)**:  
Net current into any node is zero → $Ax = 0$ for current vector $x$.  
Thus, **valid currents live in $N(A)$**.

**Kirchhoff’s Voltage Law (KVL)**:  
Sum of voltage drops around any loop is zero → $A^T y = 0$ for potential $y$.  
Thus, **consistent voltages live in $N(A^T)$**.

**Tree**:  
A connected graph with no cycles.  
**Property**: Incidence matrix has full row rank: $\text{rank}(A) = n - 1$.

**Cycle Rank (Cyclomatic Number)**:  
$\dim N(A) = m - (n - 1)$ = number of **independent cycles**.

**Euler’s Formula** (planar graphs):  
\[
|V| - |E| + |F| = 2
\quad\Rightarrow\quad
\text{#loops} = |E| - |V| + 1
\]
(for connected planar graphs; $F$ = faces including exterior).

---

### Subspaces of the Incidence Matrix

| Subspace | Location | Dimension (connected graph) | Interpretation |
|--------|----------|-----------------------------|----------------|
| $C(A)$ | $\mathbb{R}^n$ (nodes) | $n - 1$ | Potential differences (voltages) |
| $N(A)$ | $\mathbb{R}^m$ (edges) | $m - (n - 1)$ | Cycle currents (conserved flows) |
| $C(A^T)$ | $\mathbb{R}^m$ (edges) | $n - 1$ | Valid current patterns |
| $N(A^T)$ | $\mathbb{R}^n$ (nodes) | $1$ | Constant potential (ground) |

**Orthogonality**:  
- $C(A) \perp N(A^T)$ in $\mathbb{R}^n$  
- $C(A^T) \perp N(A)$ in $\mathbb{R}^m$

---

### Example: Network with 4 Nodes, 5 Edges

**Graph**:  
Nodes: $0,1,2,3$  
Edges: $0\to1$, $1\to2$, $2\to3$, $3\to1$, $0\to2$

**Incidence Matrix** $A$ (4×5):

\[
A = 
\begin{pmatrix}
\color{blue}{-1} & 0 & 0 & 0 & \color{blue}{-1} \\
\color{red}{+1} & \color{blue}{-1} & 0 & 0 & \color{red}{+1} \\
0 & \color{red}{+1} & \color{blue}{-1} & 0 & \color{red}{+1} \\
0 & 0 & \color{red}{+1} & \color{blue}{-1} & 0
\end{pmatrix}
\]

**Rank**: $3$ ($\text{rank}(A) = n - 1 = 3$)

**Column Space** $C(A)$:  
Dim = 3. Basis: first three columns (pivot columns).  
Represents all achievable **voltage drops**.

**Nullspace** $N(A)$:  
Dim = $5 - 3 = 2$.  
Basis vectors correspond to two **independent cycles**:  
- Cycle $1\to2\to3\to1$  
- Cycle $0\to1\to2\to0$ (via $0\to2$)

**Row Space** $C(A^T)$:  
Dim = 3. Basis: first three rows of $A$.  
Valid **current injection patterns** at nodes.

**Left Nullspace** $N(A^T)$:  
Dim = $4 - 3 = 1$.  
Basis: $\begin{pmatrix} 1 \\ 1 \\ 1 \\ 1 \end{pmatrix}^T$  
→ All nodes can be grounded to same potential.

---

### Kirchhoff + Ohm’s Law → Network Equations

Let:
- $x \in \mathbb{R}^m$: edge **currents**
- $y \in \mathbb{R}^n$: node **potentials**
- $C$: diagonal conductance matrix (Ohm’s law: $x = C A^T y$)
- $f \in \mathbb{R}^n$: external current sources at nodes

Then:

1. **KCL**: $A x = f$  
2. **Ohm**: $x = C A^T y$  
3. Substitute:  
   \[
   A (C A^T) y = f
   \quad\Rightarrow\quad
   \underbrace{A C A^T}_{\text{symmetric, positive semidefinite}} y = f
   \]

This is the **Laplacian system** of network theory.

---

### Applications Beyond Electricity

| Field | Incidence Matrix Represents |
|------|------------------------------|
| **Chemistry** | Reaction stoichiometry (reagents consumed/produced) |
| **Biology** | Gene regulation (up/down-regulation by pathogens) |
| **Traffic** | Flow conservation at intersections |
| **Hydraulics** | Pressure drops and flow balance |

---

### Summary Table

| Concept | Matrix Equation | Physical Meaning |
|-------|------------------|------------------|
| **KCL** | $A x = 0$ | Current conservation at nodes |
| **KVL** | $A^T y = 0$ | Voltage drop around loops = 0 |
| **Ohm** | $x = C A^T y$ | Current ∝ conductance × voltage drop |
| **Network Solve** | $A C A^T y = f$ | Solve for node voltages given sources |

---

**Key Takeaway**:  
The **incidence matrix** turns graph topology into linear algebra. The **four subspaces** directly encode:
- **Connectivity** (rank = $n-1$)
- **Cycles** (nullspace dimension)
- **Conservation laws** (KCL/KVL)
- **Solvable systems** (via $A C A^T$)

Interactive Problem Generator

This code generates a random directed graph with 4 nodes and 4–6 edges, ensuring connectivity. It computes the incidence matrix, rank, and bases for the four fundamental subspaces, with a graph visualization. Predict the rank and subspace dimensions, then check the output.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from IPython.display import Markdown, display

def generate_graph(n=4):
    """
    Generate a random directed graph with n nodes and 4-6 edges, ensuring connectivity.
    Return incidence matrix and edge list.
    """
    m = np.random.randint(4, 7)
    edges = []
    # Ensure connectivity with a path 0->1->2->3
    for i in range(n-1):
        edges.append((i, i+1))
    # Add random edges
    while len(edges) < m:
        i, j = np.random.choice(n, 2, replace=False)
        if i != j and (i, j) not in edges and (j, i) not in edges:
            edges.append((i, j))
    
    A = np.zeros((n, m))
    for k, (i, j) in enumerate(edges):
        A[i, k] = -1  # Outgoing
        A[j, k] = 1   # Incoming
    
    return A, edges

def rref(A):
    """
    Compute RREF of A and return pivot columns.
    """
    A = A.astype(float)
    m, n = A.shape
    R = A.copy()
    pivot_cols = []
    row = 0
    for col in range(n):
        if row >= m:
            break
        pivot_row = row
        while pivot_row < m and abs(R[pivot_row, col]) < 1e-10:
            pivot_row += 1
        if pivot_row < m:
            if pivot_row != row:
                R[[row, pivot_row]] = R[[pivot_row, row]]
            R[row] /= R[row, col]
            for i in range(m):
                if i != row:
                    R[i] -= R[i, col] * R[row]
            pivot_cols.append(col)
            row += 1
    return R, pivot_cols

def find_subspaces(A):
    """
    Compute bases for C(A), N(A), C(A^T), N(A^T).
    """
    R, pivot_cols = rref(A)
    rank = len(pivot_cols)
    c_a_basis = A[:, pivot_cols] if pivot_cols else np.zeros((A.shape[0], 1))
    
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    n_a_basis = Vt[rank:, :].T if rank < A.shape[1] else np.zeros((A.shape[1], 1))
    
    R_at, pivot_cols_at = rref(A.T)
    c_at_basis = A.T[:, pivot_cols_at] if pivot_cols_at else np.zeros((A.shape[0], 1))
    
    U_at, S_at, Vt_at = np.linalg.svd(A.T, full_matrices=True)
    n_at_basis = Vt_at[rank:, :].T if rank < A.shape[0] else np.zeros((A.shape[0], 1))
    
    return rank, c_a_basis, n_a_basis, c_at_basis, n_at_basis

def matrix_to_latex(M):
    rows = [r" & ".join([f"{x:.2f}" if abs(x) > 1e-10 else "0" for x in row]) for row in M]
    return r"\begin{pmatrix} " + r" \\ ".join(rows) + r"\end{pmatrix}"

def plot_graph(edges, n=4):
    """
    Plot the directed graph using networkx.
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, arrowsize=20)
    plt.title("Directed Graph")
    plt.show()

# Generate graph and incidence matrix
A, edges = generate_graph()
rank, c_a_basis, n_a_basis, c_at_basis, n_at_basis = find_subspaces(A)

# Display results
markdown = f"**Directed Graph (nodes labeled 0 to 3)**:\n\n"
display(Markdown(markdown))
plot_graph(edges)
markdown = f"**Incidence Matrix A**:\n$$\n{matrix_to_latex(A)}\n$$\n\n"
markdown += f"Predict:\n- What is the rank of A?\n- What are dim(C(A)), dim(N(A)), dim(C(A^T)), dim(N(A^T))?\n- What do solutions in N(A) represent (e.g., cycles)?\n- What does N(A^T) represent (e.g., potentials)?\n\n"
display(Markdown(markdown))

# Reveal answers (students uncomment after predicting)
# markdown = f"**Answers**:\n"
# markdown += f"- Rank of A: {rank}\n"
# markdown += f"- dim(C(A)) = {rank}, dim(N(A)) = {A.shape[1] - rank}\n"
# markdown += f"- dim(C(A^T)) = {rank}, dim(N(A^T)) = {A.shape[0] - rank}\n"
# markdown += f"- C(A) Basis:\n$$\n{matrix_to_latex(c_a_basis)}\n$$\n"
# markdown += f"- N(A) Basis:\n$$\n{matrix_to_latex(n_a_basis)}\n$$\n"
# markdown += f"- C(A^T) Basis:\n$$\n{matrix_to_latex(c_at_basis)}\n$$\n"
# markdown += f"- N(A^T) Basis:\n$$\n{matrix_to_latex(n_at_basis)}\n$$\n"
# markdown += f"- N(A) represents flows around cycles in the graph.\n"
# markdown += f"- N(A^T) represents equal potentials across nodes.\n\n"
# display(Markdown(markdown))
```