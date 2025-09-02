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
# Week 1: Matrix Review

## Lecture 1: The Geometry of Linear Equations

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/J7DzL2_Na80"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

## Summary
This lecture introduces linear algebra through the lens of solving systems of equations. We can view matrices two ways, either through the "row picture" (geometric intersection of lines/planes defined by each row-equation) and the "column picture" (linear combinations of vectors corresponding to each column).

## Key Concepts
### Row Picture
Each row from the matrix form represents a line (in 2D) or plane (in 3D). Solutions occur at the intersections of these structures.
Below is an interactive visualization of the row picture for a 2x2 system of equations (ax + by = c and dx + ey = f). Launch a live code session with Binder and change the coefficients to observe how the lines and their intersection point (solution) change!

> Question for understanding: will you be able to get a solution [X, Y] for **any point** in the plane, if the given rows are parallel lines? Hint: parallel lines don't intersect!

```{code-cell} python
:tags: [thebe]

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

# Try changing these values!
row1 = [1.0, .5, 2.0]  # [a1, b1, c1]
row2 = [3.0, 2.0, 1.0]  # [a2, b2, c2]

x = np.linspace(-5, 5, 200)

def plot_row_picture(row1, row2):
    a1, b1, c1 = row1
    a2, b2, c2 = row2
   
    fig, ax = plt.subplots(figsize=(6,6))
   
    # Line 1
    if b1 != 0:
        y1 = (c1 - a1*x)/b1
        ax.plot(x, y1, 'b-', label=f'{a1:.1f}x + {b1:.1f}y = {c1:.1f}')
    else:
        ax.axvline(c1/a1 if a1 != 0 else 0, color='b',
                   label=f'x = {(c1/a1):.1f}' if a1 != 0 else 'No line')
   
    # Line 2
    if b2 != 0:
        y2 = (c2 - a2*x)/b2
        ax.plot(x, y2, 'r-', label=f'{a2:.1f}x + {b2:.1f}y = {c2:.1f}')
    else:
        ax.axvline(c2/a2 if a2 != 0 else 0, color='r',
                   label=f'x = {(c2/a2):.1f}' if a2 != 0 else 'No line')
   
    # Intersection
    A = np.array([[a1, b1], [a2, b2]])
    b_vec = np.array([c1, c2])
    det = np.linalg.det(A)
   
    if det != 0:
        sol = np.linalg.solve(A, b_vec)
        ax.plot(sol[0], sol[1], 'ko', markersize=8, label=f'Solution: ({sol[0]:.2f},{sol[1]:.2f})')
        sol_text = f"Solution: x = {sol[0]:.2f}, y = {sol[1]:.2f}"
    else:
        sol_text = "No unique solution (parallel or coincident lines)"
   
    # Plot formatting
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axhline(0, color='k', alpha=0.3)
    ax.axvline(0, color='k', alpha=0.3)
    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Row Picture: Intersection of Lines')
    ax.legend()
    plt.show()
    display(Markdown(matrix_md))

# Call the function with the defined rows
plot_row_picture(row1, row2)
```

### Column Picture

As a completely equivalent alternative, we can split the matrix into its column vectors. Our geometric intuition here is rather different. In this form, the solutions for x and y are the proper linear combination of column vectors that are equivalent to the desired solution vector. Again, each column vector represents a vector in the space; a solution to the problem is the right combination of the column vectors (e.g., you find a combination of x=.5 * vector 1 and y=2.75 * vector 2 are equivalent to the desired vector)

> Question for understanding: if your column vectors are parallel or lie on the same line, can you compose a vector that is **off** that parallel line in terms of the given column vectors? Try it in the code below! How is this equivalent to our earlier question in row picture form?


```{code-cell} python
:tags: [thebe]
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

# Try changing these values!
col1 = [1.0, -.5]  # First column of A: [a1, a2]
col2 = [.5, 2.0]  # Second column of A: [b1, b2]
b_vec = [2.0, 1.0]  # Target vector: [c1, c2]

def plot_column_picture(col1, col2, b_vec):
    # Extract coefficients
    a1, a2 = col1
    b1, b2 = col2
    c1, c2 = b_vec
    
    # Define matrix A
    A = np.array([[a1, b1], [a2, b2]])
    
    # Solve for x, y (coefficients for column vectors)
    det = np.linalg.det(A)
    if det != 0:
        sol = np.linalg.solve(A, b_vec)
        x, y = sol
        sol_text = f"Solution: x = {x:.2f}, y = {y:.2f}"
    else:
        sol_text = "No unique solution (columns are linearly dependent)"
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot column vectors (reference, saturated)
    ax.arrow(0, 0, a1, a2, color='b', width=0.1, head_width=0.3, head_length=0.4,
             label=f'Column 1: [{a1:.1f}, {a2:.1f}]', alpha=1.0)
    ax.arrow(0, 0, b1, b2, color='r', width=0.1, head_width=0.3, head_length=0.4,
             label=f'Column 2: [{b1:.1f}, {b2:.1f}]', alpha=1.0)
    
    # Plot target vector b
    ax.arrow(0, 0, c1, c2, color='g', width=0.1, head_width=0.3, head_length=0.4,
             label=f'Target: [{c1:.1f}, {c2:.1f}]')
    
    # Plot linear combination if solution exists (more transparent)
    if det != 0:
        # Plot x * col1 from origin, tip at (x*a1, x*a2)
        ax.arrow(0, 0, x*a1, x*a2, color='b', width=0.1, head_width=0.3, head_length=0.4,
                 label=f'{x:.2f} * Column 1', alpha=0.5)
        # Plot y * col2 from (x*a1, x*a2) to (c1, c2), tip at (c1, c2)
        ax.arrow(x*a1, x*a2, c1 - x*a1, c2 - x*a2, color='r', width=0.1, head_width=0.3, head_length=0.4,
                 label=f'{y:.2f} * Column 2', alpha=0.5)
        # Solution point
        ax.plot(c1, c2, 'ko', markersize=8, label=f'Solution point: ({c1:.2f}, {c2:.2f})')
    
    # Plot formatting
    max_val = max(np.abs([a1, a2, b1, b2, c1, c2])) * 1.5
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.axhline(0, color='k', alpha=0.3)
    ax.axvline(0, color='k', alpha=0.3)
    ax.grid(True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Column Picture: Linear Combination of Columns')
    ax.legend()
    plt.show()
    display(Markdown(matrix_md))

# Call the function with the defined columns and target vector
plot_column_picture(col1, col2, b_vec)
```

## Lecture 2: Elimination with matrices

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/QVKj3LADCnA"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

## Key Concepts
### Gaussian Elimination
A systematic process to solve \( Ax = b \) by performing **row operations**:

1. Swap two rows.  
2. Multiply a row by a nonzero scalar.  
3. Add or subtract a multiple of one row to another.  

The goal is to convert \( A \) to an **upper triangular matrix** \( U \), then solve \( Ux = c \) by back-substitution.

For example, consider the system

$$
\begin{cases}
x + 2y + z = 9 \\
2x - y + 3z = 13 \\
-x + y + 2z = 3
\end{cases}
$$

We can write it in **augmented matrix form**:

$$
\left[
\begin{array}{ccc|c}
1 & 2 & 1 & 9 \\
2 & -1 & 3 & 13 \\
-1 & 1 & 2 & 3
\end{array}
\right]
$$

Perform **row operations** to eliminate variables below the diagonal:

$$
R_2 \leftarrow R_2 - 2R_1
$$

$$
R_3 \leftarrow R_3 + R_1
$$

$$
\left[
\begin{array}{ccc|c}
1 & 2 & 1 & 9 \\
0 & -5 & 1 & -5 \\
0 & 3 & 3 & 12
\end{array}
\right]
$$

Next, eliminate the element below the pivot in column 2:

$$
R_3 \leftarrow R_3 + \frac{3}{5} R_2
$$

$$
\left[
\begin{array}{ccc|c}
1 & 2 & 1 & 9 \\
0 & -5 & 1 & -5 \\
0 & 0 & \frac{18}{5} & 9
\end{array}
\right]
$$

Finally, perform **back-substitution** to solve for \( x, y, z \):

Back-substitution:

$$
z = \frac{9}{18/5} = \frac{9 \cdot 5}{18} = \frac{45}{18} = \frac{5}{2}
$$

$$
y = \frac{-5 - 1 \cdot z}{-5} = \frac{-5 - 5/2}{-5} = \frac{-10/2 - 5/2}{-5} = \frac{-15/2}{-5} = \frac{15/2}{5} = \frac{15}{10} = \frac{3}{2}
$$

$$
x = 9 - 2y - z = 9 - 2 \cdot \frac{3}{2} - \frac{5}{2} = 9 - 3 - 2.5 = 3.5 = \frac{7}{2}
$$

### Pivots:
The **pivot** is the first non-zero entry in a row after elimination. Pivots indicate the variables we solve for in each step. For example, consider:

$$
A = 
\begin{bmatrix}
2 & 1 & -1 \\
-3 & -1 & 2 \\
-1 & 1 & 2
\end{bmatrix}
$$

After the first elimination step to zero out the first column below the pivot:

$$
\begin{bmatrix}
\color{blue}{2} & 1 & -1 \\
0 & -0.5 & 0.5 \\
0 & 1.5 & 1.5
\end{bmatrix}
$$

Here, the **pivot in the first row** is highlighted in **blue**.

### Elimination Matrices:
Eliminate the first column below the pivot using  

$$
R_2 \leftarrow R_2 + \frac{3}{2} R_1
$$  

corresponds to  

$$
E_1 = 
\begin{bmatrix}
1 & 0 & 0 \\
3/2 & 1 & 0 \\
1/2 & 0 & 1
\end{bmatrix}, 
\quad E_1 A = A^{(1)}
$$

2. Eliminate the second column below the second pivot using  

$$
R_3 \leftarrow R_3 - 3 R_2
$$  

corresponds to  

$$
E_2 = 
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & -3 & 1
\end{bmatrix}, 
\quad E_2 A^{(1)} = U
$$

Thus, the overall elimination is  

$$
E_2 E_1 A = U
$$

### Permutation Matrices
Sometimes the next pivot is **zero**, and we need to move a row to continue elimination. This is done via a **permutation matrix** \( P \), which acts on the left:

$$
PA =
\begin{bmatrix}
0 & 1 & 0 \\
1 & 0 & 0 \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
0 & 2 & 1 \\
3 & 1 & -1 \\
1 & -1 & 2
\end{bmatrix}
=
\begin{bmatrix}
3 & 1 & -1 \\
0 & 2 & 1 \\
1 & -1 & 2
\end{bmatrix}
$$

> Note: Row permutations are **left-multiplications**, column permutations are **right-multiplications**.

### Composing an Elimination Matrix

We can combine multiple elimination steps into a **single matrix** \(E\) using the associative property of matrix multiplication:

$$
E_1 (E_2 A) = (E_1 E_2) A
$$

Thus, we can define a single elimination matrix:

$$
E = E_1 E_2
$$

Using the example above, applying \(E_1\) and then \(E_2\) to \(A\) is equivalent to applying \(E\) to \(A\):

$$
E A = (E_1 E_2) A = E_1 (E_2 A)
$$

This allows us to **combine all elimination steps** into one matrix, simplifying both notation and computation.

### Breakdowns:
Elimination fails if a pivot is zero and no row swap can fix it, indicating the matrix is **singular** (non-invertible) or the system has **no or infinitely many solutions**.

For example, consider the system:

$$
\begin{cases}
x + y + z = 3 \\
2x + 2y + 2z = 6 \\
x + y + z = 4
\end{cases}
$$

Augmented matrix form:

$$
\left[
\begin{array}{ccc|c}
1 & 1 & 1 & 3 \\
2 & 2 & 2 & 6 \\
1 & 1 & 1 & 4
\end{array}
\right]
$$

Attempting Gaussian elimination:

- Eliminate the first column below the pivot:

$$
R_2 \leftarrow R_2 - 2R_1, \quad
R_3 \leftarrow R_3 - R_1
$$

$$
\left[
\begin{array}{ccc|c}
1 & 1 & 1 & 3 \\
0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1
\end{array}
\right]
$$

Notice the last row corresponds to \(0x + 0y + 0z = 1\), which is impossible.  

**Conclusion:** This system has **no solution**, and the elimination "breaks down" due to a zero pivot that cannot be swapped to fix the inconsistency.


## Lecture 3: Multiplication and inverse matrices

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/FX4C-JpTFgY"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

## Lecture 4: Factorization into A = LU

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/MsIvs_6vC38"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>