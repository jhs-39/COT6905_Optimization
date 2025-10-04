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

The column space and nullspace are fundamental subspaces of a matrix A (m x n):

- **Column Space C(A)**: The set of all linear combinations of the columns of A, i.e., the span of the columns. It is a subspace of R^m. The equation Ax = b has a solution if and only if b is in C(A).
  - Dimension: The rank of A (number of linearly independent columns).
  - Example: For $A = \begin{pmatrix} 1 & 3 \\ 2 & 1 \\ 4 & 1 \end{pmatrix}$, the columns are $[1,2,4]^T$ and $[3,1,1]^T$ -> C(A) is a plane in R^3.

- **Nullspace N(A)**: The set of all x in R^n such that Ax = 0. It is a subspace of R^n.
  - Dimension: n - rank(A) (from the rank-nullity theorem).
  - Example: For the same A, solve $Ax = 0$. If $rank(A) = 2$, N(A) is a line in R^2.


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
    v1 = A[:, 0]
    v2 = A[:, 1]
    
    # Check rank
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
    
    # Add legend
    ax.legend()
    
    plt.title('Column Vectors and Column Space of A')
    plt.show()
    
    # Display matrix and rank
    def matrix_to_latex(M):
        rows = [r" & ".join([f"{x:.2f}" if abs(x) > 1e-10 else "0" for x in row]) for row in M]
        return r"\begin{pmatrix} " + r" \\ ".join(rows) + r" \end{pmatrix}"
    
    markdown = f"**Matrix A**:\n\n$$   {matrix_to_latex(A)}   $$\n\n"
    markdown += f"**Column Space Rank**: {rank}\n\n"
    display(Markdown(markdown))

# Example matrix from the section
A = np.array([[1, 3], [2, 1], [4, 1]], dtype=float)
plot_column_space(A)

# Try a rank 1 matrix
A_rank1 = np.array([[1, 2], [2, 4], [3, 6]], dtype=float)  # Columns are multiples
# plot_column_space(A_rank1)
```


## Lecture 7: Solving Ax = 0: pivot variables, special solutions

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/VqP2tREMvt0"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

### Summary

## Lecture 8: Solving Ax = b: row reduced form R

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/9Q1q7s1jTzU"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

### Summary

