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

# Week 2: Spaces

## Lecture 5: Transposes, permutations, spaces R^n

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/JibVXBElKL0"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

### Definitions
Recall: we can define a matrix P that exchanges order of rows (permutation). They fit into last lecture's matrix factorization as:
$
PA = LU
$
where P is permutation matrix, A is our target matrix, L is the lower triangular inverse elimination matrix, and U is the upper triangular row echelon form. A permutation matrix has a special property: its transpose is its inverse.
$
P^{-1} = P^{T}
$

Recall: the transpose. Turn the first row into the first column, second row into second column...
$
(A^T)_{ij} = A{ji}
$
It also reverses order when distributed, similar to inverse:
$
(R^TR)^T = R^TR^{TT} = R^TR
$

Recall: the symmetric matrix. Any matrix where $A^T = A$. This means, effectively, that it must be square (n x n), and that $A_{ij} = A_{ji}$. Note: we can construct a symmetric matrix out of a rectangular matrix (m x n). $A^T A$ is always symmetric.

Space: a set of vectors that are closed under vector addition and scalar multiplication. That is, any vector addition or scalar multiplication in that space must produce another vector in the space. It also must include the 0 vector.

Example: the space $R^2$ forms our x-y plane

Negative example: the positive quadrant of the x-y plane is NOT a space. Subtract $\begin{pmatrix} 1 \\ 1 \end{pmatrix} - \begin{pmatrix}2 \\2 \end{pmatrix} = \begin{pmatrix}-1 \\ -1 \end{pmatrix}$ -> we can enter the negative quadrant!

How do we properly define a subspace?
A subspace must also be closed under vector addition and scalar multiplication. The result is that the subspace MUST include the origin of the original space (otherwise, multiplication by scalar 0 will move us out of the set, violating closure of the subspace).

So for $R^2$, the family of possible subspaces are:
1. All of $R^2$ (you can define a subspace as the entire space)
2. Any line in $R^2$ that includes $[0,0]$
3. The zero vector

So how do we apply to matrices? Recall our column picture of the matrix -- each column is a vector.

$
A = \begin{pmatrix} 1 & 3 \\ 2 & 1 \\ 4 & 1 \end{pmatrix}
$

In the case above, the space formed by the vectors and ALL their linear combinations forms the **column space** C(A)

Start and run the code below for a visual! Adjust the matrix values to see how the plane and column vectors respond.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import Markdown, display

def plot_column_space(A=None):
    """
    Plot the column vectors and column space of a 3x2 matrix in 3D.
    """
    if A is None:
        A = np.array([[1, 3], [2, 1], [4, 1]], dtype=float)
    
    # Extract column vectors
    v1 = A[:, 0]  # First column
    v2 = A[:, 1]  # Second column
    
    # Check rank to confirm column space is a plane
    rank = np.linalg.matrix_rank(A)
    if rank != 2:
        display(Markdown(f"**Warning**: Matrix rank is {rank}. Column space is {'a line' if rank == 1 else 'degenerate'}. Expected rank 2 for a plane."))
    
    # Create figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot origin
    ax.scatter([0], [0], [0], color='black', s=50, label='Origin')
    
    # Plot column vectors as arrows
    ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='blue', label='Column 1', linewidth=2)
    ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='red', label='Column 2', linewidth=2)
    
    # Plot column space (plane) if rank is 2
    if rank == 2:
        # Generate points for the plane: x*v1 + y*v2
        x = np.linspace(-1, 1, 20)
        y = np.linspace(-1, 1, 20)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = X[i, j] * v1 + Y[i, j] * v2
                Z[i, j] = point[2]
                X[i, j] = point[0]
                Y[i, j] = point[1]
        ax.plot_surface(X, Y, Z, color='green', alpha=0.5, label='Column Space (Plane)')
    
    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    max_range = np.max(np.abs(A)) * 1.5
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    # Add legend (note: plot_surface may not appear in legend)
    ax.legend()
    
    plt.title('Column Vectors and Column Space of A')
    plt.show()
    
    # Display matrix and rank
    def matrix_to_latex(M):
        rows = [r" & ".join([f"{x:.2f}" if abs(x) > 1e-10 else "0" for x in row]) for row in M]
        return r"\begin{pmatrix} " + r" \\ ".join(rows) + r" \end{pmatrix}"
    
    markdown = f"**Matrix A**:\n\n$$ {matrix_to_latex(A)} $$\n\n"
    markdown += f"**Column Space Rank**: {rank}\n\n"
    display(Markdown(markdown))

# Example matrix
A = np.array([[1, 3], [2, 1], [4, 1]], dtype=float)
plot_column_space(A)
```

## Lecture 6: Column space and nullspace

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/8o5Cmfpeo6g"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

Recall: vector space is a set where the addition of two vectors in the set and also multiplication by scalars results in another vector in the set.

In $R^3$, a plane through the origin would be a subspace.

The union of two spaces is not necessarily a space. The intersection of two spaces is a space.

The column space and nullspace are two of the fundamental subspaces of a matrix A (m x n):

- **Column Space C(A)**: The set of all linear combinations of the columns of A, i.e., the span of the columns. It is a subspace of R^m. The equation $Ax = b$ has a solution if and only if $b$ is in $C(A)$.
  - Dimension: The rank of A (number of linearly independent columns).
  - Example: For $A = \begin{pmatrix} 1 & 3 \\ 2 & 1 \\ 4 & 1 \end{pmatrix}$, the columns are $[1,2,4]^T$ and $[3,1,1]^T$ -> $C(A)$ is a plane in $R^3$.

- **Nullspace N(A)**: The set of all vectors x in $R^n$ such that $Ax = 0$. It is a subspace of $R^n$.
  - Dimension: $n - rank(A)$ (from the rank-nullity theorem).
  - Example: For the same A, solve $Ax = 0$. If $rank(A) = 2$, $N(A)$ is a line in $R^2$


To practice, use this problem generator to create random matrices and determine the dimensions of the column space C(A) and nullspace N(A). For a 3x2 matrix:

```{code-cell} python
def generate_matrix(case='full_rank'):
    """
    Generate a random 3x2 matrix for the specified case.
    """
    if case == 'full_rank':
        # Linearly independent columns
        v1 = np.random.randint(-5, 6, 3)
        v2 = np.random.randint(-5, 6, 3)
        while np.linalg.matrix_rank(np.column_stack((v1, v2))) != 2:
            v2 = np.random.randint(-5, 6, 3)
        A = np.column_stack((v1, v2))
    elif case == 'dependent':
        # Dependent columns (v2 = scalar * v1)
        v1 = np.random.randint(-5, 6, 3)
        while np.all(v1 == 0):
            v1 = np.random.randint(-5, 6, 3)
        scalar = np.random.choice([-2, -1, 2, 3])
        v2 = scalar * v1
        A = np.column_stack((v1, v2))
    else:
        raise ValueError("Case must be 'full_rank' or 'dependent'")
    
    return A

def plot_column_space_and_nullspace(A):
    """
    Plot the column vectors, column space, and nullspace (if non-trivial) in 3D.
    Returns the LaTeX string for the matrix and nullspace basis.
    """
    v1 = A[:, 0]
    v2 = A[:, 1]
    
    rank = np.linalg.matrix_rank(A)
    
    # 3D plot for column space and nullspace
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter([0], [0], [0], color='black', s=50, label='Origin')
    
    # Plot column vectors
    ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='blue', label='Column 1', linewidth=2)
    ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='red', label='Column 2', linewidth=2)
    
    # Plot column space
    if rank == 2:
        x = np.linspace(-1, 1, 20)
        y = np.linspace(-1, 1, 20)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = X[i, j] * v1 + Y[i, j] * v2
                Z[i, j] = point[2]
                X[i, j] = point[0]
                Y[i, j] = point[1]
        ax.plot_surface(X, Y, Z, color='green', alpha=0.5, label='Column Space (Plane)')
    elif rank == 1:
        t = np.linspace(-1, 1, 20)
        line_x = t * v1[0]
        line_y = t * v1[1]
        line_z = t * v1[2]
        ax.plot(line_x, line_y, line_z, color='green', label='Column Space (Line)', linewidth=2)
    
    # Compute nullspace basis using SymPy
    A_sym = Matrix(A)
    nullspace_basis = A_sym.nullspace()
    
    # Convert SymPy vectors to NumPy for plotting
    nullspace_basis_np = [np.array(vec).astype(float).flatten() for vec in nullspace_basis]
    
    # Plot nullspace in 3D (if non-trivial)
    nullspace_latex = ""
    if rank == 1:
        n1 = nullspace_basis_np[0]
        t = np.linspace(-2, 2, 20)
        # Nullspace vectors are in R^2, plot with z=0 in 3D
        line_x = t * n1[0]
        line_y = t * n1[1]
        line_z = np.zeros_like(t)  # z=0 for nullspace vectors
        ax.plot(line_x, line_y, line_z, color='purple', label='Nullspace (Line in x1-x2 plane)', linewidth=2)
        
        # Format nullspace basis
        def vector_to_latex(v):
            return r"\begin{pmatrix} " + r" \\ ".join([str(x) for x in v]) + r" \end{pmatrix}"
        nullspace_latex = f"\n\n**Nullspace Basis**:\n\n$$ {vector_to_latex(nullspace_basis[0])} $$"
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    max_range = max(np.max(np.abs(A)) * 1.5, np.max([np.abs(n1) for n1 in nullspace_basis_np] + [0]) * 2.5)
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    ax.legend()
    
    plt.title('Column Vectors, Column Space, and Nullspace of A in R^3')
    plt.show()
    
    # Format matrix
    def matrix_to_latex(M):
        rows = [r" & ".join([f"{x:.0f}" if abs(x) > 1e-10 else "0" for x in row]) for row in M]
        return r"\begin{pmatrix} " + r" \\ ".join(rows) + r" \end{pmatrix}"
    
    return matrix_to_latex(A), nullspace_latex

def reveal_dimensions(A):
    """
    Reveal the dimensions of C(A) and N(A).
    """
    rank = np.linalg.matrix_rank(A)
    dim_c = rank
    dim_n = A.shape[1] - rank
    markdown = f"**Dimension of C(A)**: {dim_c}\n\n"
    markdown += f"**Dimension of N(A)**: {dim_n}\n\n"
    display(Markdown(markdown))

# Generate and display a full rank matrix
A_full = generate_matrix('full_rank')
display(Markdown("**Problem 1: Full Rank Matrix**"))
matrix_latex, nullspace_latex = plot_column_space_and_nullspace(A_full)
display(Markdown(f"**Matrix A**:\n\n$$ {matrix_latex} $$\n\n{nullspace_latex}\n\n"))
display(Markdown("Predict the dimensions of C(A) and N(A). Since rank is 2, expect a plane for C(A) and trivial N(A). Uncomment reveal_dimensions(A_full) to check."))

# Generate and display a dependent matrix
A_dep = generate_matrix('dependent')
display(Markdown("**Problem 2: Dependent Columns Matrix**"))
matrix_latex, nullspace_latex = plot_column_space_and_nullspace(A_dep)
display(Markdown(f"**Matrix A**:\n\n$$ {matrix_latex} $$\n\n{nullspace_latex}\n\n"))
display(Markdown("Predict the dimensions of C(A) and N(A). Since rank is 1, expect a line for C(A) and a line for N(A) in the x1-x2 plane at z=0. Uncomment reveal_dimensions(A_dep) to check."))

# To reveal answers (students uncomment after predicting)
# reveal_dimensions(A_full)
# reveal_dimensions(A_dep)
```


## Lecture 7: Solving Ax = 0: pivot variables, special solutions

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/VqP2tREMvt0"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

We explore solving the homogeneous system $Ax = 0$ to find the nullspace $N(A)$. We introduce pivot variables, free variables, and special solutions to describe the structure of the nullspace, using the row reduced echelon form (RREF).

### Key Definitions

**Nullspace** $N(A)$: All $x \in \mathbb{R}^n$ such that $Ax = 0$, a subspace of $\mathbb{R}^n$. Dimension: $n - \text{rank}(A)$

**Row Reduced Echelon Form** (RREF): Obtained via Gaussian elimination. Contains:

1. **Pivot columns**: Have the first non-zero entry (usually 1) in each non-zero row.

2. **Free columns**: A column that does not have a pivot.

3. **Pivot Variables**: Correspond to pivot columns, determined by the system.

4. **Free Variables**: Correspond to free columns, assigned arbitrary values.

5. **Special Solutions**: For each free variable, set it to 1 (others 0) and solve for pivot variables. These form a basis for $N(A)$

Example

For $A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{pmatrix}$:

RREF: $\begin{pmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \end{pmatrix}$.

Pivot column: 1. Free columns: 2, 3.

Pivot variable: $x_1$. Free variables: $x_2$, $x_3$.

Special solutions:

Set $ x_2 = 1 $, $x_3 = 0$: $x_1 + 2x_2 = x_1 + 2 = 0 $, so $x_1 = -2 $. Solution: $\begin{pmatrix} -2 \\ 1 \\ 0 \end{pmatrix}$

Set $ x_2 = 0 $, $ x_3 = 1 $: $ x_1 + 3x_3 = x_1 + 3 = 0 $, so $x_1 = -3$. Solution: $\begin{pmatrix} -3 \\ 0 \\ 1 \end{pmatrix}$

Nullspace basis: $\begin{pmatrix} -2 \\ 1 \\ 0 \end{pmatrix}$, $\begin{pmatrix} -3 \\ 0 \\ 1 \end{pmatrix}$

Interactive Problem Generator

Use this code to generate a random 3x3 matrix with rank 2 (one free variable). It computes the RREF, identifies pivot and free variables, and finds special solutions. Predict the pivot/free variables and nullspace basis, then check the output.

```{code-cell} python
import numpy as np
from sympy import Matrix
from IPython.display import Markdown, display

def generate_matrix():
    """
    Generate a random 3x3 matrix with rank 2.
    """
    v1 = np.random.randint(-5, 6, 3)
    v2 = np.random.randint(-5, 6, 3)
    while np.linalg.matrix_rank(np.column_stack((v1, v2))) != 2:
        v2 = np.random.randint(-5, 6, 3)
    # Make third column a linear combination
    coeffs = np.random.randint(-2, 3, 2)
    v3 = coeffs[0] * v1 + coeffs[1] * v2
    A = np.column_stack((v1, v2, v3))
    return A

def find_special_solutions(A):
    """
    Find special solutions for N(A) using SymPy's RREF.
    """
    # Convert to SymPy Matrix
    A_sym = Matrix(A)
    # Compute RREF and pivot columns
    R, pivot_cols = A_sym.rref()
    R = np.array(R).astype(float)  # Convert back to NumPy for display
    m, n = A.shape
    free_cols = [i for i in range(n) if i not in pivot_cols]
    special_solutions = []
    
    for free_col in free_cols:
        x = np.zeros(n, dtype=float)
        x[free_col] = 1
        for pivot_col, row in zip(pivot_cols, range(len(pivot_cols))):
            x[pivot_col] = -float(R[row, free_col])
        special_solutions.append(x)
    
    return R, pivot_cols, free_cols, special_solutions

def matrix_to_latex(M):
    """
    Convert matrix to LaTeX, handling SymPy rationals or NumPy floats.
    """
    if isinstance(M, Matrix):
        rows = [r" & ".join([str(x) for x in row]) for row in M.tolist()]
    else:
        rows = [r" & ".join([f"{x:.2f}" if abs(x) > 1e-10 else "0" for x in row]) for row in M]
    return r"\begin{pmatrix} " + r" \\ ".join(rows) + r" \end{pmatrix}"

# Generate and analyze matrix
A = generate_matrix()
R, pivot_cols, free_cols, special_solutions = find_special_solutions(A)

# Display results
markdown = f"**Matrix A**:\n$ {matrix_to_latex(A)} $\n\n"
markdown += f"**RREF of A**:\n$ {matrix_to_latex(R)} $\n\n"
markdown += f"**Pivot Columns**: {[i+1 for i in pivot_cols]} (1-based indexing)\n\n"
markdown += f"**Free Columns**: {[i+1 for i in free_cols]} (1-based indexing)\n\n"
if special_solutions:
    markdown += "**Special Solutions (Nullspace Basis)**:\n\n"
    for i, sol in enumerate(special_solutions, 1):
        markdown += f"Solution {i}:\n$ {matrix_to_latex(sol.reshape(-1, 1))} $\n\n"
else:
    markdown += "**Nullspace**: Trivial (only the zero vector)\n\n"
display(Markdown(markdown))

# Verify solutions
for i, sol in enumerate(special_solutions, 1):
    if np.allclose(np.dot(A, sol), 0):
        display(Markdown(f"**Verification**: Solution {i} satisfies Ax = 0"))
    else:
        display(Markdown(f"**Error**: Solution {i} does not satisfy Ax = 0"))
```

## Lecture 8: Solving Ax = b: row reduced form R

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/9Q1q7s1jTzU"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

We explore solving the non-homogeneous (e.g., has nonzero solutions b) system $Ax = b$, where $A$ is an $m \times n$ matrix and $b \in \mathbb{R}^m$. Using the row reduced echelon form (RREF), we determine if solutions exist and find the complete solution: a particular solution plus the nullspace.

### Key Definitions

System $Ax = b$: Find $ x \in \mathbb{R}^n$ such that $Ax = b$. Solvable if $b \in C(A)$.

Row Reduced Echelon Form (RREF): From Gaussian elimination on $[A | b]$, yields $[R | c]$
Complete Solution: $x = x_p + x_n $, where $x_p$ is a particular solution and $x_n \in N(A)$

### Example

For $A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 5 \end{pmatrix}$, $b = \begin{pmatrix} 4 \\ 5 \end{pmatrix} $: \
Augmented matrix: $ [A | b] = \begin{pmatrix} 1 & 2 & 3 & 4 \\ 2 & 4 & 5 & 5 \end{pmatrix}$ \
RREF: $[R | c] = \begin{pmatrix} 1 & 2 & 0 & -1 \\ 0 & 0 & 1 & 3 \end{pmatrix}$ \
Pivot columns: 1, 3. Free column: 2. \
Particular solution: Set $x_2 = 0$, so $x_1 = -1$, $x_3 = 3$: $x_p = \begin{pmatrix} -1 \\ 0 \\ 3 \end{pmatrix}$
Nullspace: Solve $Rx = 0$. Free variable $x_2$. From $x_1 + 2x_2 = 0 $, $x_1 = -2x_2$ Special solution: $x_2 = 1 $, $ x_1 = -2 $, $x_3 = 0 $: $ \begin{pmatrix} -2 \\ 1 \\ 0 \end{pmatrix} $ \
Complete solution: $x = \begin{pmatrix} -1 \\ 0 \\ 3 \end{pmatrix} + s \begin{pmatrix} -2 \\ 1 \\ 0 \end{pmatrix}$


### Interactive Problem Generator

This code generates a random 3x3 matrix with rank 2 and a random $b$. It computes the RREF of $[A | b]$, checks solvability, and finds the particular and nullspace solutions. Predict if the system is consistent and describe the solution, then check the output.

```{code-cell} python
import numpy as np
from IPython.display import Markdown, display

def generate_matrix_and_b():
    """
    Generate a random 3x3 matrix with rank 2 and a consistent b in C(A).
    """
    v1 = np.random.randint(-5, 6, 3)
    v2 = np.random.randint(-5, 6, 3)
    while np.linalg.matrix_rank(np.column_stack((v1, v2))) != 2:
        v2 = np.random.randint(-5, 6, 3)
    coeffs = np.random.randint(-2, 3, 2)
    v3 = coeffs[0] * v1 + coeffs[1] * v2
    A = np.column_stack((v1, v2, v3))
    # Generate b in C(A)
    scalars = np.random.randint(-2, 3, 3)
    b = scalars[0] * v1 + scalars[1] * v2 + scalars[2] * v3
    return A, b

def rref(A, b=None):
    """
    Compute RREF of A or [A | b].
    """
    if b is not None:
        A = np.column_stack((A, b))
    A = A.astype(float)
    m, n = A.shape
    R = A.copy()
    pivot_cols = []
    row = 0
    for col in range(n if b is None else n-1):
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

def solve_ax_b(A, b):
    """
    Solve Ax = b, returning particular solution and nullspace basis.
    """
    R, pivot_cols = rref(A, b)
    m, n = R.shape
    free_cols = [i for i in range(n-1) if i not in pivot_cols]
    
    # Particular solution: set free variables to 0
    particular = np.zeros(n-1)
    for pivot_col, row in zip(pivot_cols, range(len(pivot_cols))):
        particular[pivot_col] = R[row, -1]
    
    # Nullspace solutions
    R_A, pivot_cols_A = rref(A)
    special_solutions = []
    for free_col in free_cols:
        x = np.zeros(n-1)
        x[free_col] = 1
        for pivot_col, row in zip(pivot_cols_A, range(len(pivot_cols_A))):
            x[pivot_col] = -R_A[row, free_col]
        special_solutions.append(x)
    
    return R, pivot_cols, free_cols, particular, special_solutions

def matrix_to_latex(M):
    rows = [r" & ".join([f"{x:.0f}" if abs(x) > 1e-10 else "0" for x in row]) for row in M]
    return r"\begin{pmatrix} " + r" \\ ".join(rows) + r"\end{pmatrix}"

# Generate and solve
A, b = generate_matrix_and_b()
R, pivot_cols, free_cols, particular, special_solutions = solve_ax_b(A, b)

# Display results
markdown = f"**Matrix A**:\n$ {matrix_to_latex(A)} $\n\n"
markdown += f"**Vector b**:\n$ {matrix_to_latex(b.reshape(-1, 1))} $\n\n"
markdown += f"**RREF of [A | b]**:\n$ {matrix_to_latex(R)} $\n\n"
markdown += f"**Pivot Columns**: {[i+1 for i in pivot_cols]} (1-based indexing)\n\n"
markdown += f"**Free Columns**: {[i+1 for i in free_cols]} (1-based indexing)\n\n"
markdown += "**System is Consistent** (b is in C(A))\n\n"
if particular is not None:
    markdown += f"**Particular Solution**:\n$ {matrix_to_latex(particular.reshape(-1, 1))} $\n\n"
if special_solutions:
    markdown += "**Nullspace Basis (Special Solutions)**:\n\n"
    for i, sol in enumerate(special_solutions, 1):
        markdown += f"Solution {i}:\n$ {matrix_to_latex(sol.reshape(-1, 1))} $\n\n"
else:
    markdown += "**Nullspace**: Trivial (only the zero vector)\n\n"

display(Markdown(markdown))

# Verify solutions
if particular is not None:
    if np.allclose(np.dot(A, particular), b):
        display(Markdown("**Verification**: Particular solution satisfies Ax = b"))
    else:
        display(Markdown("**Error**: Particular solution does not satisfy Ax = b"))
    for i, sol in enumerate(special_solutions, 1):
        if np.allclose(np.dot(A, sol), 0):
            display(Markdown(f"**Verification**: Nullspace solution {i} satisfies Ax = 0"))
        else:
            display(Markdown(f"**Error**: Nullspace solution {i} does not satisfy Ax = 0"))
```

