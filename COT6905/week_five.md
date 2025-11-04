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

We learn how to turn **any basis** into an **orthonormal basis** — the "best" coordinate system for projections and least squares.

---

### 1. Orthogonal and Orthonormal Vectors

Two vectors $u, v$ are **orthogonal** if  
$
u^T v = 0
$

They are **orthonormal** if also  
$
\|u\| = \|v\| = 1
$

For a set $\{q_1, \dots, q_k\}$:  
$
\boxed{q_i^T q_j = \delta_{ij} =
\begin{cases}
1 & i = j \\
0 & i \neq j
\end{cases}}
$

---

### 2. Orthogonal Matrices

Let  
$
Q = \begin{bmatrix} q_1 & \cdots & q_n \end{bmatrix} \quad \text{(square, } n \times n\text{)}
$

Then  
$
\boxed{Q^T Q = I} \quad \Rightarrow \quad \boxed{Q^T = Q^{-1}}
$

**Definition**: A square matrix with orthonormal columns is an **orthogonal matrix**.

> **Note**: Only **square** matrices can be orthogonal.

---

### 3. Examples of Orthogonal Matrices

#### a) Permutation Matrix
$
P = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}
$
Columns are standard basis vectors (permuted) → orthonormal → $P^T P = I$.

#### b) Hadamard Matrix (size 2)
$
H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}
$
Entries $\pm 1$, scaled to unit length → orthogonal.

---

### 4. Why Orthonormal Bases Are Useful

Suppose $Q$ has **orthonormal columns** (not necessarily square).

The **projection onto $C(Q)$** is  
$
P = Q (Q^T Q)^{-1} Q^T
$

But $Q^T Q = I_k$ (for first $k$ columns), so  
$
\boxed{P = Q Q^T}
$

And  
$
\boxed{P^2 = (Q Q^T)(Q Q^T) = Q (Q^T Q) Q^T = Q I Q^T = Q Q^T = P}
$

**Projection is clean**: no inversion needed!

If $Q$ is square:  
$
P = Q Q^T = I \quad \Rightarrow \quad C(Q) = \mathbb{R}^n
$

---

### 5. Least Squares with Orthonormal Columns

Recall:  
$
A^T A \hat{x} = A^T b \quad \Rightarrow \quad \hat{x} = (A^T A)^{-1} A^T b
$

If $A = Q$ (orthonormal columns):  
$
\boxed{\hat{x} = Q^T b}
\quad \text{(no inversion!)}
$

---

### 6. Gram-Schmidt: Constructing Orthonormal Bases

Start with linearly independent vectors $\{a_1, a_2, \dots\}$.

**Goal**: Build $\{q_1, q_2, \dots\}$ orthonormal.

#### Step 1: First Vector
$
v_1 = a_1, \quad
\boxed{q_1 = \frac{v_1}{\|v_1\|}}
$

#### Step 2: Second Vector
Subtract projection onto $q_1$:  
$
v_2 = a_2 - (q_1^T a_2) q_1, \quad
\boxed{q_2 = \frac{v_2}{\|v_2\|}}
$

#### Step 3: General Step
$
v_k = a_k - \sum_{j=1}^{k-1} (q_j^T a_k) q_j, \quad
q_k = \frac{v_k}{\|v_k\|}
$

**Guaranteed**: $v_k \perp \text{span}\{q_1, \dots, q_{k-1}\}$

---

### 7. QR Decomposition

For any $m \times n$ matrix $A$ with independent columns:  
$
\boxed{A = Q R}
$
- $Q$: $m \times n$, orthonormal columns  
- $R$: $n \times n$, **upper triangular**

Like $A = LU$, but with **orthonormal $Q$**.

---

### 8. Example: Gram-Schmidt in $\mathbb{R}^3$

Let  
$
a_1 = \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}, \;
a_2 = \begin{pmatrix} 1 \\ 1 \\ 0 \end{pmatrix}, \;
a_3 = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}
$

**By hand**:

1. $v_1 = a_1$, $\quad \|v_1\| = \sqrt{3}$, $\quad q_1 = \frac{1}{\sqrt{3}} \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}$

2. $v_2 = a_2 - (q_1^T a_2) q_1 = \begin{pmatrix}1\\1\\0\end{pmatrix} - \frac{2}{\sqrt{3}} \cdot \frac{1}{\sqrt{3}} \begin{pmatrix}1\\1\\1\end{pmatrix} = \begin{pmatrix}1/3 \\ 1/3 \\ -2/3\end{pmatrix}$  
   $\|v_2\| = 1$, $\quad q_2 = \begin{pmatrix} 1/3 \\ 1/3 \\ -2/3 \end{pmatrix}$

3. $v_3 = a_3 - (q_1^T a_3)q_1 - (q_2^T a_3)q_2 = \cdots = \begin{pmatrix} 1/3 \\ -2/3 \\ 1/3 \end{pmatrix}$  
   $\|v_3\| = 1$, $\quad q_3 = \begin{pmatrix} 1/3 \\ -2/3 \\ 1/3 \end{pmatrix}$

---

### Interactive Problem Generator

This code generates a random 3x2 matrix A (rank 2), applies Gram-Schmidt to find an orthonormal basis for C(A), and forms an orthogonal matrix Q. It visualizes the original and orthonormal basis vectors in 3D. Predict the orthonormal basis and verify orthogonality, then check the output.

```{code-cell} python
# ------------------------------------------------------------------
#  Lecture 17 – Gram-Schmidt: 3 orthonormal vectors in R^3
# ------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import Markdown, display

# --------------------------------------------------------------
# 1. Generate a random 3×3 matrix with full rank
# --------------------------------------------------------------
def generate_full_rank_3x3():
    """Return a 3×3 matrix with linearly independent columns."""
    while True:
        A = np.random.randint(-5, 6, size=(3, 3))
        if np.linalg.matrix_rank(A) == 3:
            return A.astype(float)

A = generate_full_rank_3x3()   # ← change seed or re-run for new example

# --------------------------------------------------------------
# 2. Gram-Schmidt (returns Q only – R is optional)
# --------------------------------------------------------------
def gram_schmidt(A):
    A = A.astype(float)
    m, n = A.shape
    Q = np.zeros((m, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            v -= (Q[:, i] @ A[:, j]) * Q[:, i]
        Q[:, j] = v / np.linalg.norm(v)
    return Q

Q = gram_schmidt(A)

# --------------------------------------------------------------
# 3. LaTeX helper
# --------------------------------------------------------------
def matrix_to_latex(M, name="A", dec=2):
    rows = []
    for row in M:
        entries = [f"{x:.{dec}f}" if abs(x) > 1e-8 else "0" for x in row]
        rows.append(" & ".join(entries))
    body = r" \\ ".join(rows)
    return f"${name} = \\begin{{pmatrix}} {body} \\end{{pmatrix}}$"

# --------------------------------------------------------------
# 4. 3-D quiver plot
# --------------------------------------------------------------
def plot_orthonormal_basis(A, Q):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Original Basis (blue/red/yellow) → Orthonormal Basis (green/purple/orange)')

    colors_orig = ['blue', 'red', 'gold']
    colors_orth = ['green', 'purple', 'orange']
    labels_orig = [f'$a_{i+1}$' for i in range(3)]
    labels_orth = [f'$q_{i+1}$' for i in range(3)]

    # Original vectors
    for i in range(3):
        ax.quiver(0,0,0, A[0,i], A[1,i], A[2,i],
                  color=colors_orig[i], linewidth=2.5, arrow_length_ratio=0.1,
                  label=labels_orig[i])

    # Orthonormal vectors
    for i in range(3):
        ax.quiver(0,0,0, Q[0,i], Q[1,i], Q[2,i],
                  color=colors_orth[i], linewidth=2.5, arrow_length_ratio=0.1,
                  label=labels_orth[i], alpha=0.9)

    ax.set_xlim([-6,6]); ax.set_ylim([-6,6]); ax.set_zlim([-6,6])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# --------------------------------------------------------------
# 5. Verification
# --------------------------------------------------------------
ortho_check = np.allclose(Q.T @ Q, np.eye(3), atol=1e-6)
unit_check  = np.allclose(np.linalg.norm(Q, axis=0), 1.0, atol=1e-6)

# --------------------------------------------------------------
# 6. Display: Hand-work section
# --------------------------------------------------------------
markdown = f"""
### **Your Task: Find 3 Orthonormal Vectors by Hand**

**Given basis vectors** (columns of $A$):  

{matrix_to_latex(A.T, name="a_1, a_2, a_3", dec=1)}

**Step-by-step (do this on paper!)**:

1. $q_1 = \\dfrac{{a_1}}{{\\|a_1\\|}}$  
2. $v_2 = a_2 - (q_1^T a_2) q_1$,&nbsp;&nbsp; $q_2 = \\dfrac{{v_2}}{{\\|v_2\\|}}$  
3. $v_3 = a_3 - (q_1^T a_3) q_1 - (q_2^T a_3) q_2$,&nbsp;&nbsp; $q_3 = \\dfrac{{v_3}}{{\\|v_3\\|}}$

**Predict**: What are $q_1, q_2, q_3$? Is $Q^T Q = I$?

---

*Now run the cell and compare!*
"""

display(Markdown(markdown))
plot_orthonormal_basis(A, Q)

# --------------------------------------------------------------
# 7. Reveal answers
# --------------------------------------------------------------
reveal = f"""
**Computer Solution**:

**Orthonormal basis** $Q$:

{matrix_to_latex(Q, name="Q", dec=3)}

**Checks**:  
- $Q^T Q = I$? → **{ortho_check}**  
- All columns unit length? → **{unit_check}**

*Great job if your hand calculation matches (up to sign)!*
"""

display(Markdown(reveal))
```

## Lecture 18: Properties of determinants

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/srxexLishgY"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

We now turn to **square matrices** and introduce the **determinant** — a single number that tells us whether a matrix is invertible, and by how much it scales area (or volume).

---

### 1. What Is the Determinant?

For a square matrix $A \in \mathbb{R}^{n \times n}$,  
$$
\boxed{\det(A) \text{ or } |A|}
$$
is a **scalar** with these key properties:

| Property | Meaning |
|--------|-------|
| $\det(A) \neq 0$ | $A$ is **invertible** |
| $\det(A) = 0$ | $A$ is **singular** (no inverse) |

---

### 2. The 4 Defining Properties of the Determinant

These **completely define** $\det$:

1. **Identity**:  
   $$
   \boxed{\det(I) = 1}
   $$

2. **Row Swap**:  
   Swapping two rows **reverses the sign**:  
   $$
   \det(\text{swap rows}) = -\det(A)
   $$
   → Permutation matrices have $\det(P) = \pm 1$ (even/odd).

3. **Linearity (Two Parts)**:  
   - **Scaling a row**:  
     $$
     \det(\text{row}_1 \leftarrow t \cdot \text{row}_1) = t \cdot \det(A)
     $$
   - **Adding rows**:  
     $$
     \det(\text{row}_1 \leftarrow \text{row}_1 + \text{row}_2) = \det(A) + \det(\text{swapped version})
     $$

4. **Duplicate Rows**:  
   $$
   \text{If two rows are identical} \quad \Rightarrow \quad \boxed{\det(A) = 0}
   $$

---

### 3. Elimination Preserves the Determinant!

Elimination (without swapping) **does not change** $\det(A)$:

$$
\det\begin{pmatrix} a & b \\ c - \ell a & d - \ell b \end{pmatrix}
= \det\begin{pmatrix} a & b \\ c & d \end{pmatrix} + \det\begin{pmatrix} a & b \\ -\ell a & -\ell b \end{pmatrix}
= \det(A) + (-\ell) \det(A) = \det(A)
$$

**Key**: Only **scaling** or **swapping** changes $\det$.

---

### 4. Triangular Matrices: The Easy Case

For **upper triangular** $U$:  
$$
\boxed{\det(U) = \text{product of diagonal entries}}
$$

**Proof via elimination**:  
Elimination turns any invertible $A$ into $U$ (with possible row swaps).  
$\det(A) = (\pm 1) \cdot \det(U) = \pm d_1 d_2 \cdots d_n$

---

### 5. When Is $\det(A) = 0$?

From elimination:  
$$
\boxed{\det(A) = 0 \quad \Leftrightarrow \quad \text{a row of zeros appears during elimination}}
\quad \Leftrightarrow \quad A \text{ is singular}
$$

---

### 6. The 2×2 Formula (You Already Know)

$$
\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc
$$

This is the **only formula you need to memorize** — the rest follows from properties!

---

### 7. Multiplicative Property

$$
\boxed{\det(AB) = \det(A) \det(B)}
$$

**Downstream consequences**:

| Result | Proof |
|------|------|
| $\det(A^{-1}) = \frac{1}{\det(A)}$ | $\det(A^{-1} A) = \det(I) = 1$ |
| $\det(A^k) = [\det(A)]^k$ | $\det(A \cdot A \cdots A)$ |
| $\det(tA) = t^n \det(A)$ | Scale each row by $t$ |
| $\det(A^T) = \det(A)$ | $A^T = L^T U^T$, $\det(L^T) = \det(L)$, etc. |

---

### 8. Example: 3×3 Determinant via Elimination

Let  
$$
A = \begin{pmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 1 & 0 & 6 \end{pmatrix}
$$

Eliminate (no swaps):  
$$
\rightarrow \begin{pmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 0 & -2 & 3 \end{pmatrix}
\rightarrow \begin{pmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 0 & 0 & 3.5 \end{pmatrix}
$$

$$
\det(A) = 1 \cdot 4 \cdot 3.5 = 14
$$

**Solving Determinants using LU Decomposition**

```{code-cell} python
# ------------------------------------------------------------------
#  Lecture 18 – Determinant via Elimination (LU Decomposition)
# ------------------------------------------------------------------
import numpy as np
from scipy.linalg import lu
from IPython.display import Markdown, display

# Change this matrix to explore!
A = np.array([[1, 2, 3],
              [0, 4, 5],
              [1, 0, 6]], dtype=float)

# LU decomposition (without pivoting for simplicity)
P, L, U = lu(A)  # P is permutation, L lower, U upper

# Determinant = product of diagonal of U, times det(P) = ±1
det_U = np.prod(np.diag(U))
det_P = np.linalg.det(P)  # +1 or -1
det_computed = det_P * det_U
det_numpy = np.linalg.det(A)

# LaTeX helper
def matrix_to_latex(M, name="A", dec=1):
    rows = [r" & ".join([f"{x:.{dec}f}" if abs(x) > 1e-8 else "0" for x in row]) for row in M]
    return f"${name} = \\begin{{pmatrix}} {r' \\ '.join(rows)} \\end{{pmatrix}}$"

summary = f"""
### Determinant via Elimination

**Matrix** $A$:  
{matrix_to_latex(A)}

**LU Decomposition** (with permutation $P$):  
$P = {matrix_to_latex(P, 'P', dec=0)}$  
$L = {matrix_to_latex(L, 'L')}$  
$U = {matrix_to_latex(U, 'U')}$

**Determinant** = $\det(P) \cdot \text{prod}(\text{diag}(U))$  
$= {det_P:+.0f} \cdot {det_U:.3f} = {det_computed:.3f}$

**NumPy check**: $\det(A) = {det_numpy:.3f}$ → **match!**

*Try changing $A$ and re-run — watch how row swaps flip the sign!*
"""

display(Markdown(summary)
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