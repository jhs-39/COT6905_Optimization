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

## Lecture 15: Projections onto subspaces

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/Y_Ac6KiQ1t0"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>

## Lecture 16: Projection Matrices

<iframe width="560" height="315"
    src="https://www.youtube.com/embed/osh80YCg_GM"
    frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
    allowfullscreen>
</iframe>
