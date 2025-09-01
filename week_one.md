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
:::{thebe-button}
:::
(launch:thebe)=
# Week 1: Matrix Review

## Lecture 1: The Geometry of Linear Equations

:::{iframe} https://www.youtube.com/embed/J7DzL2_Na80
:width: 100%
Lecture 1
:::

Jake notes (to remove): In this section we need to communicate row and column pictures [ ]; intuition for solving systems of equations in both ways [ ]; introduce foundations for invertibility in high dim spaces [ ] (even if we don't use that language yet)

## Summary
This lecture introduces linear algebra through the lens of solving systems of equations. We can view matrices two ways, either through the "row picture" (geometric intersection of lines/planes defined by each row-equation) and the "column picture" (linear combinations of vectors corresponding to each column).

## Key Concepts
- **Row Picture**: Each equation represents a line (in 2D) or plane (in 3D), corresponding to a row of the matrix. Solutions are the intersections of these structures.
Below is an interactive visualization of the row picture for a 2x2 system of equations (ax + by = c and dx + ey = f). Adjust the sliders to change the coefficients and observe how the lines and their intersection point (solution) change. The matrix form of the system is also shown.

```{code-cell} ipython3
:tags: [hide-input, thebe]
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
import warnings
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

def plot_row_picture(a1=2.0, b1=1.0, c1=5.0, a2=1.0, b2=-1.0, c2=1.0):
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
      
    # Define x range for plotting
    x = np.linspace(-5, 5, 100)
      
    # Plot line 1: ax + by = c
    if b1 != 0:
        y1 = (c1 - a1 * x) / b1
        ax.plot(x, y1, 'b-', label=f'{a1:.1f}x + {b1:.1f}y = {c1:.1f}')
    else:
        ax.axvline(c1/a1 if a1 != 0 else 0, color='b', label=f'x = {(c1/a1):.1f}' if a1 != 0 else 'No line')
      
    # Plot line 2: dx + ey = f
    if b2 != 0:
        y2 = (c2 - a2 * x) / b2
        ax.plot(x, y2, 'r-', label=f'{a2:.1f}x + {b2:.1f}y = {c2:.1f}')
    else:
        ax.axvline(c2/a2 if a2 != 0 else 0, color='r', label=f'x = {(c2/a2):.1f}' if a2 != 0 else 'No line')
      
    # Calculate intersection
    A = np.array([[a1, b1], [a2, b2]])
    b = np.array([c1, c2])
    det = np.linalg.det(A)
      
    if det != 0:
        sol = np.linalg.solve(A, b)
        x_sol, y_sol = sol
        ax.plot(x_sol, y_sol, 'ko', markersize=8, label=f'Solution: ({x_sol:.2f}, {y_sol:.2f})')
        print(f"Solution: x = {x_sol:.2f}, y = {y_sol:.2f}")
    else:
        print("No unique solution (parallel or coincident lines)")
      
    # Set plot properties
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.axhline(0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(0, color='k', linestyle='-', alpha=0.3)
    ax.grid(True)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Row Picture: Intersection of Lines')
    plt.show()
      
    # Display matrix form
    print(f"Matrix form:\n$$\\begin{{bmatrix}} {a1:.1f} & {b1:.1f} \\\\ {a2:.1f} & {b2:.1f} \\end{{bmatrix}} \\begin{{bmatrix}} x \\\\ y \\end{{bmatrix}} = \\begin{{bmatrix}} {c1:.1f} \\\\ {c2:.1f} \\end{{bmatrix}}$$")

  interact(plot_row_picture,
           a1=FloatSlider(min=-5, max=5, step=0.1, value=2.0, description='a'),
           b1=FloatSlider(min=-5, max=5, step=0.1, value=1.0, description='b'),
           c1=FloatSlider(min=-10, max=10, step=0.1, value=5.0, description='c'),
           a2=FloatSlider(min=-5, max=5, step=0.1, value=1.0, description='d'),
           b2=FloatSlider(min=-5, max=5, step=0.1, value=-1.0, description='e'),
           c2=FloatSlider(min=-10, max=10, step=0.1, value=1.0, description='f'))
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

:::{iframe} https://www.youtube.com/embed/QVKj3LADCnA
:width: 100%
Lecture 2
:::

## Lecture 3: Multiplication and inverse matrices

:::{iframe} https://www.youtube.com/embed/FX4C-JpTFgY
:width: 100%
Lecture 3
:::

## Lecture 4: Factorization into A = LU

:::{iframe} https://www.youtube.com/embed/MsIvs_6vC38
:width: 100%
Lecture 4
:::