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

# Week 5: Determinants

## Lecture 17: Orthogonal matrices and Gram-Schmidt

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/0MtwqhIwdrI"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

This lecture (inspired by Gilbert Strang’s MIT 18.06 Lecture 17) introduces orthogonal matrices and the Gram-Schmidt process, key for creating orthonormal bases and simplifying computations. It builds on Lecture 14’s orthogonality and Lecture 15’s projections, connecting to optimization in your syllabus.

### Key Definitions and Geometric Intuitions

Orthogonal Matrix:

Definition: A square matrix ( Q ) (n x n) with orthonormal columns: ( Q^T Q = I ), so ( Q^{-1} = Q^T ).

Geometric Intuition: Represents rotations or reflections, preserving lengths (( |Qx| = |x| )) and angles. Columns form an orthonormal coordinate system in ( \mathbb{R}^n ).

Gram-Schmidt Process:

Definition: Transforms a basis ( {v_1, \ldots, v_k} ) into an orthonormal basis ( {q_1, \ldots, q_k} ):
( u_1 = v_1 ), ( q_1 = \frac{u_1}{|u_1|} ).
( u_i = v_i - \sum_{j=1}^{i-1} (q_j^T v_i) q_j ), ( q_i = \frac{u_i}{|u_i|} ).
Geometric Intuition: Orthogonalizes vectors by subtracting projections onto previous vectors, creating a perpendicular basis. Normalization ensures unit lengths.
Applications:
Simplifies projections: ( p = \sum (q_i^T b) q_i ).
Enables QR factorization: ( A = QR ), where ( Q ) is orthogonal.
Example
For ( A = \begin{pmatrix} 1 & 0 \ 1 & 1 \ 0 & 1 \end{pmatrix} ):
Gram-Schmidt on columns ( v_1 = \begin{pmatrix} 1 \ 1 \ 0 \end{pmatrix} ), ( v_2 = \begin{pmatrix} 0 \ 1 \ 1 \end{pmatrix} ):
( q_1 = \frac{v_1}{|v_1|} = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \ 1 \ 0 \end{pmatrix} ).
( u_2 = v_2 - (q_1^T v_2) q_1 = \begin{pmatrix} 0 \ 1 \ 1 \end{pmatrix} - \frac{1}{\sqrt{2}} \cdot \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \ 1 \ 0 \end{pmatrix} = \begin{pmatrix} -0.5 \ 0.5 \ 1 \end{pmatrix} ), ( q_2 = \frac{u_2}{|u_2|} ).
Orthogonal matrix: ( Q = \begin{pmatrix} \frac{1}{\sqrt{2}} & \frac{-1}{\sqrt{6}} \ \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{6}} \ 0 & \frac{2}{\sqrt{6}} \end{pmatrix} ).

### Interactive Problem Generator

This code generates a random 3x2 matrix ( A ) (rank 2), applies Gram-Schmidt to find an orthonormal basis for ( C(A) ), and forms an orthogonal matrix ( Q ). It visualizes the original and orthonormal basis vectors in 3D. Predict the orthonormal basis and verify orthogonality, then check the output.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import Markdown, display

def generate_matrix():
    """
    Generate a random 3x2 matrix with rank 2.
    """
    v1 = np.random.randint(-5, 6, 3)
    v2 = np.random.randint(-5, 6, 3)
    while np.linalg.matrix_rank(np.column_stack((v1, v2))) != 2:
        v2 = np.random.randint(-5, 6, 3)
    A = np.column_stack((v1, v2))
    return A

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

def matrix_to_latex(M):
    rows = [r" & ".join([f"{x:.2f}" if abs(x) > 1e-10 else "0" for x in row]) for row in M]
    return r"\begin{pmatrix} " + r" \\ ".join(rows) + r"\end{pmatrix}"

def plot_vectors(A, Q):
    """
    Plot original and orthonormal basis vectors in 3D.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot origin
    ax.scatter([0], [0], [0], color='black', s=50, label='Origin')
    
    # Plot original basis
    ax.quiver(0, 0, 0, A[0, 0], A[1, 0], A[2, 0], color='blue', label='Original v1', linewidth=2)
    ax.quiver(0, 0, 0, A[0, 1], A[1, 1], A[2, 1], color='red', label='Original v2', linewidth=2)
    
    # Plot orthonormal basis
    ax.quiver(0, 0, 0, Q[0, 0], Q[1, 0], Q[2, 0], color='green', label='Orthonormal q1', linewidth=2)
    ax.quiver(0, 0, 0, Q[0, 1], Q[1, 1], Q[2, 1], color='purple', label='Orthonormal q2', linewidth=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    max_range = np.max(np.abs(np.concatenate([A, Q], axis=1))) * 1.5
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    ax.legend()
    plt.title('Original and Orthonormal Basis for C(A)')
    plt.show()

# Generate matrix
A = generate_matrix()
Q = gram_schmidt(A)

# Verify orthogonality
ortho_check = np.allclose(np.dot(Q.T, Q), np.eye(Q.shape[1]))
length_check = np.allclose(np.linalg.norm(Q, axis=0), 1)

# Display results
markdown = f"**Matrix A (basis for C(A))**:\n$$\n{matrix_to_latex(A)}\n$$\n\n"
display(Markdown(markdown))
plot_vectors(A, Q)
markdown = f"Predict:\n- Orthonormal basis for C(A) using Gram-Schmidt.\n- Is Q^T Q = I?\n- Are columns of Q unit length?\n\n"
display(Markdown(markdown))

# Reveal answers (students uncomment after predicting)
# markdown = f"**Answers**:\n"
# markdown += f"- Orthonormal Basis Q:\n$$\n{matrix_to_latex(Q)}\n$$\n"
# markdown += f"- Q^T Q = I: {'True' if ortho_check else 'False'}.\n"
# markdown += f"- Columns unit length: {'True' if length_check else 'False'}.\n\n"
# display(Markdown(markdown))
```

## Lecture 18: Properties of determinants

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/srxexLishgY"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

This lecture explores the determinant of a square matrix, its properties, and its geometric interpretation as a volume scaling factor. It builds on matrix theory concepts, connecting to invertibility and linear systems.

### Key Definitions and Geometric Intuitions

Determinant:

Definition: For an ( n \times n ) matrix ( A ), ( \det(A) ) is a scalar indicating invertibility and volume scaling.

Geometric Intuition: ( |\det(A)| ) is the volume of the parallelepiped formed by ( A )’s columns. If ( \det(A) = 0 ), the columns are linearly dependent, collapsing the space.

Properties:

Invertibility: ( A ) is invertible if ( \det(A) \neq 0 ).

Row Operations:

Swap rows: ( \det(A') = -\det(A) ).

Multiply row by ( k ): ( \det(A') = k \det(A) ).

Add multiple of one row to another: ( \det(A') = \det(A) ).

Multiplicative: ( \det(AB) = \det(A) \det(B) ).

Transpose: ( \det(A^T) = \det(A) ).

Triangular: ( \det(A) = \text{product of diagonal entries} ).

Zero/Dependent Rows: ( \det(A) = 0 ).

Applications:
Test invertibility.

Compute volumes in geometry.

Solve systems via Cramer’s rule.

Example

For ( A = \begin{pmatrix} 1 & 2 & 0 \ 0 & 3 & 1 \ 2 & 0 & 4 \end{pmatrix} ):

Determinant: ( \det(A) = 1 \cdot (3 \cdot 4 - 1 \cdot 0) - 2 \cdot (0 \cdot 4 - 1 \cdot 2) + 0 \cdot (0 \cdot 0 - 3 \cdot 2) = 12 + 4 = 16 ).

Row swap (rows 1 and 2): ( A' = \begin{pmatrix} 0 & 3 & 1 \ 1 & 2 & 0 \ 2 & 0 & 4 \end{pmatrix} ), ( \det(A') = -16 ).

Multiply row 1 by 2: ( A'' = \begin{pmatrix} 2 & 4 & 0 \ 0 & 3 & 1 \ 2 & 0 & 4 \end{pmatrix} ), ( \det(A'') = 2 \cdot 16 = 32 ).

Interactive Problem Generator

This code generates a random 3x3 matrix, computes its determinant, and applies row operations (swap, scale, add). It visualizes the parallelepiped formed by the columns in 3D to show the volume. Predict the determinant and effects of row operations, then check the output.

```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import Markdown, display

def generate_matrix():
    """
    Generate a random 3x3 matrix with integer entries.
    """
    A = np.random.randint(-5, 6, (3, 3))
    while np.abs(np.linalg.det(A)) < 1e-10:  # Avoid singular matrices
        A = np.random.randint(-5, 6, (3, 3))
    return A

def apply_row_operations(A):
    """
    Apply row operations: swap, scale, add.
    Return modified matrices and determinants.
    """
    # Swap rows 0 and 1
    A_swap = A.copy()
    A_swap[[0, 1]] = A_swap[[1, 0]]
    
    # Scale row 0 by 2
    A_scale = A.copy()
    A_scale[0] *= 2
    
    # Add 2 * row 1 to row 2
    A_add = A.copy()
    A_add[2] += 2 * A_add[1]
    
    return A_swap, A_scale, A_add

def matrix_to_latex(M):
    rows = [r" & ".join([f"{x:.0f}" if abs(x) > 1e-10 else "0" for x in row]) for row in M]
    return r"\begin{pmatrix} " + r" \\ ".join(rows) + r"\end{pmatrix}"

def plot_parallelepiped(A):
    """
    Plot the parallelepiped formed by A’s columns in 3D.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Origin
    ax.scatter([0], [0], [0], color='black', s=50, label='Origin')
    
    # Column vectors
    v1, v2, v3 = A[:, 0], A[:, 1], A[:, 2]
    ax.quiver(0, 0, 0, v1[0], v1[1], v1[2], color='blue', label='Column 1', linewidth=2)
    ax.quiver(0, 0, 0, v2[0], v2[1], v2[2], color='red', label='Column 2', linewidth=2)
    ax.quiver(0, 0, 0, v3[0], v3[1], v3[2], color='green', label='Column 3', linewidth=2)
    
    # Parallelepiped edges
    points = np.array([
        [0, 0, 0], v1, v2, v3,
        v1 + v2, v1 + v3, v2 + v3, v1 + v2 + v3
    ])
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (2, 4), (2, 6),
        (3, 5), (3, 6), (1, 5),
        (4, 7), (5, 7), (6, 7)
    ]
    for edge in edges:
        ax.plot(points[edge, 0], points[edge, 1], points[edge, 2], 'k-')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    max_range = np.max(np.abs(A)) * 1.5
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    ax.legend()
    plt.title('Parallelepiped Formed by Columns of A')
    plt.show()

# Generate matrix and apply operations
A = generate_matrix()
A_swap, A_scale, A_add = apply_row_operations(A)
det_A = np.linalg.det(A)
det_swap = np.linalg.det(A_swap)
det_scale = np.linalg.det(A_scale)
det_add = np.linalg.det(A_add)

# Display results
markdown = f"**Matrix A**:\n$$\n{matrix_to_latex(A)}\n$$\n\n"
display(Markdown(markdown))
plot_parallelepiped(A)
markdown = f"Predict:\n- det(A).\n- det(A) after swapping rows 0 and 1.\n- det(A) after scaling row 0 by 2.\n- det(A) after adding 2 * row 1 to row 2.\n- Is A invertible?\n\n"
display(Markdown(markdown))

# Reveal answers (students uncomment after predicting)
# markdown = f"**Answers**:\n"
# markdown += f"- det(A): {det_A:.2f}\n"
# markdown += f"- det after swap: {det_swap:.2f} (should be -det(A))\n"
# markdown += f"- det after scale: {det_scale:.2f} (should be 2 * det(A))\n"
# markdown += f"- det after add: {det_add:.2f} (should equal det(A))\n"
# markdown += f"- A invertible: {'Yes' if abs(det_A) > 1e-10 else 'No'}.\n\n"
# display(Markdown(markdown))
```

## Lecture 19: Determinant formulas and cofactors

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/23LLB9mNJvc"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

## Lecture 20: Cramer's rule, inverse matrix, and volume

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/QNpj-gOXW9M"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>