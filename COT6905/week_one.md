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

Jake notes (to remove): In this section we need to communicate row and column pictures [ ]; intuition for solving systems of equations in both ways [ ]; introduce foundations for invertibility in high dim spaces [ ] (even if we don't use that language yet)

## Summary
This lecture introduces linear algebra through the lens of solving systems of equations. We can view matrices two ways, either through the "row picture" (geometric intersection of lines/planes defined by each row-equation) and the "column picture" (linear combinations of vectors corresponding to each column).

## Key Concepts
- **Row Picture**: Each equation represents a line (in 2D) or plane (in 3D), corresponding to a row of the matrix. Solutions are the intersections of these structures.
Below is an interactive visualization of the row picture for a 2x2 system of equations (ax + by = c and dx + ey = f). Adjust the sliders to change the coefficients and observe how the lines and their intersection point (solution) change. The matrix form of the system is also shown.

```{code-cell} python
:tags: [thebe]

import numpy as np
import plotly.graph_objects as go
from IPython.display import display, Markdown

# Initial slider ranges
a1_vals = np.linspace(-5,5,11)
b1_vals = np.linspace(-5,5,11)
c1_vals = np.linspace(-10,10,11)
a2_vals = np.linspace(-5,5,11)
b2_vals = np.linspace(-5,5,11)
c2_vals = np.linspace(-10,10,11)

# Initial values
a1, b1, c1 = 2.0, 1.0, 5.0
a2, b2, c2 = 1.0, -1.0, 1.0
x = np.linspace(-5,5,200)

def make_figure(a1,b1,c1,a2,b2,c2):
    fig = go.Figure()
    
    # Line 1
    if b1 != 0:
        y1 = (c1 - a1*x)/b1
        fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name=f'{a1:.1f}x+{b1:.1f}y={c1:.1f}', line=dict(color='blue')))
    else:
        fig.add_trace(go.Scatter(x=[c1/a1]*2, y=[-5,5], mode='lines', name=f'x={c1/a1:.1f}', line=dict(color='blue')))
    
    # Line 2
    if b2 != 0:
        y2 = (c2 - a2*x)/b2
        fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name=f'{a2:.1f}x+{b2:.1f}y={c2:.1f}', line=dict(color='red')))
    else:
        fig.add_trace(go.Scatter(x=[c2/a2]*2, y=[-5,5], mode='lines', name=f'x={c2/a2:.1f}', line=dict(color='red')))
    
    # Intersection
    A = np.array([[a1,b1],[a2,b2]])
    b_vec = np.array([c1,c2])
    det = np.linalg.det(A)
    
    if det != 0:
        sol = np.linalg.solve(A,b_vec)
        fig.add_trace(go.Scatter(x=[sol[0]], y=[sol[1]], mode='markers', marker=dict(color='black', size=10), name=f'Solution ({sol[0]:.2f},{sol[1]:.2f})'))
        sol_text = f"Solution: x={sol[0]:.2f}, y={sol[1]:.2f}"
    else:
        sol_text = "No unique solution (parallel or coincident lines)"
    
    # Layout
    fig.update_layout(
        title='Row Picture: Intersection of Lines',
        xaxis=dict(range=[-5,5], zeroline=True, zerolinecolor='gray'),
        yaxis=dict(range=[-5,5], zeroline=True, zerolinecolor='gray'),
        width=600,
        height=600
    )
    
    fig.show()
    
    # Display LaTeX matrix
    matrix_md = f"""
**{sol_text}**

Matrix form:

$$
\\begin{{bmatrix}} {a1:.1f} & {b1:.1f} \\\\ {a2:.1f} & {b2:.1f} \\end{{bmatrix}}
\\begin{{bmatrix}} x \\\\ y \\end{{bmatrix}} =
\\begin{{bmatrix}} {c1:.1f} \\\\ {c2:.1f} \\end{{bmatrix}}
$$
"""
    display(Markdown(matrix_md))

# Use Plotly sliders
fig = go.FigureWidget()
fig.update_layout(width=600, height=600)
fig.show()

```

- **Column Picture**: Rewrite as x * column1 + y * column2 = b, finding coefficients for vector combinations.

Each column represents a vector; a solution to the problem is the proper combination of vectors. 

- Sneak peek: invertibility. If multiple points in a high dimensional space are transformed into the same point in low-dimemnsional space, can the reverse be performed?

## Tutorial: Solving a 2x2 System
Consider the system:
- 2x + y = 5
- x - y = 1

**Step 1 (Elimination)**: Add the equations to eliminate y: 3x = 6 → x = 2.  
**Step 2**: Substitute: 2(2) + y = 5 → y = 1.  

In the row picture: Lines intersect at (2,1).  
In the column picture: 2 * [2,1]^T + 1 * [1,-1]^T = [5,1]^T.

### Python Example (Using NumPy)
```python
import numpy as np

# Define matrix A and vector b
A = np.array([[2, 1], [1, -1]])
b = np.array([5, 1])

# Solve Ax = b
x = np.linalg.solve(A, b)
print("Solution: x =", x[0], ", y =", x[1])
```

## Lecture 2: Elimination with matrices

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/QVKj3LADCnA"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

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