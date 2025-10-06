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

**Dimension**: The number of vectors in a basis. For a matrix ( A ) (m x n):
$\dim(C(A)) = \text{rank}(A)$
$\dim(N(A)) = n - \text{rank}(A)$
**Rank-Nullity Theorem**: $\text{rank}(A) + \dim(N(A)) = n $

Example

For $A = \begin{pmatrix} 1 & 2 & 2 \\ 3 & 4 & 6 \\ 5 & 6 & 10 \end{pmatrix} $: \

RREF: $\begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \\ 0 & 0 & 0 \end{pmatrix}$\
Rank: 2 (two pivot columns). \
Columns 1 and 2 are independent (pivot columns). Column 3 is dependent (e.g., $2 v_1 + v_2 - v_3 = 0 $) \
Basis for $C(A)$: $\left{ \begin{pmatrix} 1 \\ 3 \\ 5 \end{pmatrix}, \begin{pmatrix} 2 \\ 4 \\ 6 \end{pmatrix}} $ \
Dimensions: $\dim(C(A)) = 2 $, $ \dim(N(A)) = 3 - 2 = 1 $ \

Interactive Problem Generator

This code generates two 3x3 matrices:

Rank 2: Two independent columns.

Rank 1: All columns dependent (multiples of one vector). For each, compute the RREF and predict:

Are the columns linearly independent?

What is a basis for $C(A)$?

What are $\dim(C(A))$ and $\dim(N(A))$? Check the output to verify.

```{code-cell} python
import numpy as np
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

def rref(A):
    """
    Compute RREF of A.
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

def matrix_to_latex(M):
    rows = [r" & ".join([f"{x:.0f}" if abs(x) > 1e-10 else "0" for x in row]) for row in M]
    return r"\begin{pmatrix} " + r" \\ ".join(rows) + r"\end{pmatrix}"

# Generate and analyze matrices
A_rank2 = generate_matrix('rank2')
R_rank2, pivot_cols_rank2 = rref(A_rank2)
A_rank1 = generate_matrix('rank1')
R_rank1, pivot_cols_rank1 = rref(A_rank1)

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
# rank = np.linalg.matrix_rank(A_rank2)
# markdown = f"**Answers for Problem 1**:\n"
# markdown += f"- Linear Independence: Columns {pivot_cols_rank2} are independent; others dependent.\n"
# markdown += f"- Basis for C(A): Columns {[i+1 for i in pivot_cols_rank2]} of A.\n"
# markdown += f"- dim(C(A)) = {rank}, dim(N(A)) = {A_rank2.shape[1] - rank}.\n\n"
# display(Markdown(markdown))
#
# rank = np.linalg.matrix_rank(A_rank1)
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
**Row Space** $C(A^T)$: Subspace of $\mathbb{R}^n$, spanned by $A$’s rows $columns of ( A^T )$. Dimension: $\text{rank}(A)$ \
**Left Nullspace** $N(A^T)$: Subspace of $\mathbb{R}^m$, all $y$ such that $A^T y = 0$. Dimension: $m - \text{rank}(A)$ \
**Orthogonality**: $C(A) \perp N(A^T)$ in $ \mathbb{R}^m $, $C(A^T) \perp N(A)$ in $\mathbb{R}^n$ \
**Rank-Nullity**: $\text{rank}(A) + \dim(N(A)) = n $, $\text{rank}(A^T) + \dim(N(A^T)) = m$ \

Example

For $A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{pmatrix} $ (rank 1): \
RREF: $\begin{pmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \end{pmatrix} ) $ \
Column Space: $C(A) = \text{span}{ \begin{pmatrix} 1 \\ 2 \end{pmatrix} }$, dim = 1 \
Nullspace: Basis $\left{ \begin{pmatrix} -2 \\ 1 \\ 0 \end{pmatrix}$, $\begin{pmatrix} -3 \\ 0 \\ 1 \end{pmatrix}}$, $dim = 3 - 1 = 2 $ \
Row Space: $C(A^T) = \text{span}{ \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix} }$, $dim = 1$ \
Left Nullspace: Basis $\begin{pmatrix} -2 \\ 1 \end{pmatrix}$, $dim = 2 - 1 = 1$ \


```{code-cell} python
import numpy as np
from IPython.display import Markdown, display

def generate_matrix(case='rank2'):
    """
    Generate a random 3x3 matrix with specified rank.
    """
    if case == 'rank2':
        v1 = np.random.randint(-5, 6, 3)
        v2 = np.random.randint(-5, 6, 3)
        while np.linalg.matrix_rank(np.column_stack((v1, v2))) != 2:
            v2 = np.random.randint(-5, 6, 3)
        coeffs = np.random.randint(-2, 3, 2)
        v3 = coeffs[0] * v1 + coeffs[1] * v2
        A = np.column_stack((v1, v2, v3))
    elif case == 'rank1':
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
    # Column space and nullspace of A
    R, pivot_cols = rref(A)
    rank = len(pivot_cols)
    c_a_basis = A[:, pivot_cols]  # Pivot columns of A
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    n_a_basis = Vt[rank:, :].T  # Nullspace basis
    
    # Row space and left nullspace (from A^T)
    R_at, pivot_cols_at = rref(A.T)
    c_at_basis = A.T[:, pivot_cols_at]  # Pivot columns of A^T (rows of A)
    U_at, S_at, Vt_at = np.linalg.svd(A.T, full_matrices=True)
    n_at_basis = Vt_at[rank:, :].T  # Left nullspace basis
    
    return rank, c_a_basis, n_a_basis, c_at_basis, n_at_basis

def matrix_to_latex(M):
    rows = [r" & ".join([f"{x:.0f}" if abs(x) > 1e-10 else "0" for x in row]) for row in M]
    return r"\begin{pmatrix} " + r" \\ ".join(rows) + r"\end{pmatrix}"

# Generate and analyze matrices
A_rank2 = generate_matrix('rank2')
rank2, c_a_basis2, n_a_basis2, c_at_basis2, n_at_basis2 = find_subspaces(A_rank2)
A_rank1 = generate_matrix('rank1')
rank1, c_a_basis1, n_a_basis1, c_at_basis1, n_at_basis1 = find_subspaces(A_rank1)

# Display Problem 1: Rank 2
markdown = f"**Problem 1: Rank 2 Matrix**\n\n"
markdown += f"**Matrix A**:\n$ {matrix_to_latex(A_rank2)} $\n\n"
markdown += f"Predict:\n- dim(C(A)), dim(N(A)), dim(C(A^T)), dim(N(A^T))?\n- Basis for each subspace?\n\n"

display(Markdown(markdown))

# Display Problem 2: Rank 1
markdown = f"**Problem 2: Rank 1 Matrix**\n\n"
markdown += f"**Matrix A**:\n$ {matrix_to_latex(A_rank1)} $\n\n"
markdown += f"Predict:\n- dim(C(A)), dim(N(A)), dim(C(A^T)), dim(N(A^T))?\n- Basis for each subspace?\n\n"

display(Markdown(markdown))

# Reveal answers (students uncomment after predicting)
# markdown = f"**Answers for Problem 1**:\n"
# markdown += f"- dim(C(A)) = {rank2}, dim(N(A)) = {3 - rank2}, dim(C(A^T)) = {rank2}, dim(N(A^T)) = {3 - rank2}\n"
# markdown += f"- C(A) Basis:\n$ {matrix_to_latex(c_a_basis2)} $\n"
# markdown += f"- N(A) Basis:\n$ {matrix_to_latex(n_a_basis2)} $\n"
# markdown += f"- C(A^T) Basis:\n$ {matrix_to_latex(c_at_basis2)} $\n"
# markdown += f"- N(A^T) Basis:\n$ {matrix_to_latex(n_at_basis2)} $\n\n"
# display(Markdown(markdown))
#
# markdown = f"**Answers for Problem 2**:\n"
# markdown += f"- dim(C(A)) = {rank1}, dim(N(A)) = {3 - rank1}, dim(C(A^T)) = {rank1}, dim(N(A^T)) = {3 - rank1}\n"
# markdown += f"- C(A) Basis:\n$ {matrix_to_latex(c_a_basis1)} $\n"
# markdown += f"- N(A) Basis:\n$ {matrix_to_latex(n_a_basis1)} $\n"
# markdown += f"- C(A^T) Basis:\n$ {matrix_to_latex(c_at_basis1)} $\n"
# markdown += f"- N(A^T) Basis:\n$ {matrix_to_latex(n_at_basis1)} $\n\n"
# display(Markdown(markdown))
```

## Lecture 11: Matrix spaces; rank 1; small world graphs

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/2IdtqGM6KWU"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

This lecture explores matrix spaces as vector spaces, the structure of rank-1 matrices, and small world graphs via adjacency matrices. These concepts extend our understanding of subspaces and apply linear algebra to networks.

### Key Definitions
Matrix Space $M_{m \times n}$: The vector space of all $m \times n$ matrices, with dimension $m \cdot n$. Basis: matrices $E_{ij}$ (1 in position ( (i,j) ), 0 elsewhere). \

Rank-1 Matrix: A matrix $A = u v^T$, where $u \in \mathbb{R}^m$, $v \in \mathbb{R}^n$. All rows are multiples of $v^T$, columns multiples of $u$. Rank = 1 \

Small World Graphs: Graphs where nodes are connected by short paths. Adjacency matrix $A$ has $A_{ij} = 1$ if nodes $i, j$ are connected, 0 otherwise. $ (A^k)_{ij}$ counts paths of length $k$ \

Example

Rank-1 Matrix: $A = \begin{pmatrix} 1 \\ 2 \end{pmatrix} \begin{pmatrix} 3 & 4 \end{pmatrix} = \begin{pmatrix} 3 & 4 \ 6 & 8 \end{pmatrix} $. Rank = 1, rows are multiples of $\begin{pmatrix} 3 & 4 \end{pmatrix}$ \

Matrix Space: For 2x2 matrices, basis is $\left{ \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}, \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$, $\begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}, \begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}} $, $dim = 4$ \

Small World Graph: Adjacency matrix $A = \begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 0 \end{pmatrix}$. Compute $A^2$ for 2-step paths.

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
**Graph**: Nodes connected by edges (directed or undirected).
**Incidence Matrix** $A$: For a directed graph with $n$ nodes and $m$ edges:
Size $n \times m$.

Each column represents an edge $i \to j$: -1 at row $i$ (outgoing), +1 at row $j$ (incoming), 0 elsewhere.

Subspaces:

Column Space $C(A)$:
Definition: Subspace of $\mathbb{R}^n$ ($n$ = number of nodes), spanned by the columns of $A$, each representing the effect of an edge on node potentials.
Dimension: $\dim(C(A)) = \text{rank}(A)$. For a connected graph, $\text{rank}(A) = n - 1$, as the rows of $A$ sum to zero (linear dependence).
Geometric Intuition: Represents potential differences across nodes due to edge flows. Each column (e.g., $\begin{pmatrix} -1 \\ 1 \\ 0 \\ \vdots \\ 0 \end{pmatrix}$) shows a unit flow from node $i$ to $j$, affecting their potentials. $C(A)$ is a hyperplane in $\mathbb{R}^n$ (dim $n - 1$ for connected graphs), capturing all possible node potential differences induced by edge flows, like voltage drops in a circuit.
Nullspace $N(A)$:
Definition: Subspace of $\mathbb{R}^m$ ($m$ = number of edges), all vectors $x \in \mathbb{R}^m$ such that $Ax = 0$, assigning weights to edges.
Dimension: $\dim(N(A)) = m - \text{rank}(A)$. For a connected graph, $m - (n - 1)$.
Geometric Intuition: Represents flows around cycles. If $Ax = 0$, the net flow into and out of each node is zero (Kirchhoff’s current law for potentials). In $\mathbb{R}^m$, $x$ assigns weights to edges forming closed loops (cycles), like a current circulating in a triangle 1→2→3→1. The dimension reflects the number of independent cycles (graph’s cycle rank).
Row Space $C(A^T)$:
Definition: Subspace of $\mathbb{R}^m$, spanned by the rows of $A$ (columns of $A^T$), each corresponding to a node’s edge connections.
Dimension: $\dim(C(A^T)) = \text{rank}(A) = n - 1$ for a connected graph.
Geometric Intuition: Represents edge flow patterns respecting node connectivity. Each row (e.g., $\begin{pmatrix} -1 & 0 & 1 & 0 \end{pmatrix}$) encodes edges incident to a node. $C(A^T)$ is a subspace of $\mathbb{R}^m$, orthogonal to $N(A)$, capturing flow patterns that align with node incidences, such as current distributions in a network.
Left Nullspace $N(A^T)$:
Definition: Subspace of $\mathbb{R}^n$, all vectors $y \in \mathbb{R}^n$ such that $A^T y = 0$.
Dimension: $\dim(N(A^T)) = n - \text{rank}(A) = n - (n - 1) = 1$ for a connected graph.
Geometric Intuition: Represents equal potentials across nodes. If $A^T y = 0$, $y$ assigns the same potential to all nodes in a connected graph (Kirchhoff’s voltage law). In $\mathbb{R}^n$, $N(A^T)$ is a line spanned by $\begin{pmatrix} 1 \\ 1 \\ \vdots \\ 1 \end{pmatrix}$, reflecting nodes at the same potential (e.g., ground voltage in a circuit).
Rank: For a connected graph, $\text{rank}(A) = n - 1$.
Orthogonality: $C(A) \perp N(A^T)$ in $\mathbb{R}^n$, $C(A^T) \perp N(A)$ in $\mathbb{R}^m$.

Example

For a directed graph with 4 nodes, edges $0 \to 1$, $1 \to 2$, $2 \to 3$, $3 \to 1$, $0 \to 2$:
Incidence matrix: $ A = \begin{pmatrix} -1 & 0 & 0 & 0 & -1 \\ 1 & -1 & 0 & 1 & 0 \\ 0 & 1 & -1 & 0 & 1 \\ 0 & 0 & 1 & -1 & 0 \end{pmatrix} $
Rank: 3 ($n = 4$, connected).

$C(A)$: Dim = 3, basis: first 3 columns. Represents potential differences.

$N(A)$: Dim = $5 - 3 = 2$, basis includes cycle flows (e.g., 1→2→3→1).

$C(A^T)$: Dim = 3, basis: first 3 rows. Represents edge flow patterns.

$N(A^T)$: Dim = $4 - 3 = 1$, basis: $\begin{pmatrix} 1 \\ 1 \\ 1 \\ 1 \end{pmatrix}$. Represents equal node potentials.

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