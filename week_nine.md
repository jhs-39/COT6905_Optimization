# Information Theory: Introduction to Shannon Entropy

## Overview
Information theory provides tools to quantify uncertainty and information in data, which is crucial for deep learning tasks like optimization and model compression. Shannon entropy measures the average "surprise" in a probability distribution, helping us understand concepts like cross-entropy loss and KL divergence (as covered in our syllabus on probabilistic tools).

:::{note}
In deep learning, low entropy indicates confident predictions, while high entropy signals uncertainty—useful for detecting out-of-distribution data or guiding compression strategies.
:::

## Shannon Entropy Formula
For a categorical probability vector \( P = [p_1, p_2, \dots, p_n] \) where \( \sum p_i = 1 \):

:::{math}
:name: eq:shannon_entropy

H(P) = -\sum_{i=1}^n p_i \log_2 p_i
:::

We use base-2 logarithms to measure in *bits*. For \( p_i = 0 \), the term is defined as 0.

## Binary Entropy Example
Let's simplify to a binary distribution: probabilities \( p \) and \( 1-p \). The entropy is:

\[ H(p) = -p \log_2 p - (1-p) \log_2 (1-p) \]

This reaches a maximum of 1 bit at \( p = 0.5 \) (complete uncertainty) and 0 at the extremes (certainty).

## Interactive Visualization
Use the slider below to adjust \( p \) and see how entropy changes. (If not interactive, clone the repo and run locally.)

:::{code-cell} python
:tags: [remove-input]  # Hides code input for cleaner display; remove if you want visible code

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
import math  # For log, but we'll use np for safety

def plot_entropy(p=0.5):
    if p == 0 or p == 1:
        entropy = 0.0
    else:
        entropy = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(['Entropy'], [entropy], color='skyblue')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Bits')
    ax.set_title(f'Shannon Entropy: {entropy:.2f} bits')
    plt.show()

interact(plot_entropy, p=FloatSlider(min=0.0, max=1.0, step=0.01, value=0.5));
:::

Experiment: Slide p toward 0 or 1—what happens to Shannon entropy?

## Relevance to Deep Learning
In optimization (e.g., SGD variants from the syllabus), entropy helps regularize models. For example, in knowledge distillation, we minimize KL divergence, which involves entropy differences between teacher and student models.