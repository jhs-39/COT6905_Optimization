# Information Theory: Introduction to Shannon Entropy

## Overview
Information theory provides tools to quantify uncertainty and information content of data. Tools from information theory form the backbone of neural net loss functions which are used to train models. Cross-Entropy loss and KL Divergence are used extensively for this purpose, which are derived from information-theoretic formulations. Separately, many models predict outcomes using categorical probability distributions. The Shannon Entropy can be applied to these outputs to characterized how uncertain the model is about its prediction, which is important for evaluating models for risk-aware deployments. Finally, these tools can be applied during training to analyze the dynamics, by researchers at the cutting edge -- we give a brief overview of these recent contributions.

## Shannon Entropy Formula
For a categorical probability vector $P = [p_1, p_2, \dots, p_n]$ where $\sum p_i = 1$:

:::{math}
:name: eq:shannon_entropy
H(P) = -\sum_{i=1}^n p_i \log_2 p_i
:::

We use base-2 logarithms to measure in *bits*. For \( p_i = 0 \), the term is defined as 0.

## What is the Shannon Entropy? A Binary Entropy Example
Let's simplify to a binary distribution, aka a coin flip: probabilities \( p \) and \( 1-p \). The entropy is:

\[ H(p) = -p \log_2 p - (1-p) \log_2 (1-p) \]

This reaches a maximum of 1 bit at \( p = 0.5 \) (complete uncertainty) and 0 at the extremes (certainty). In simple terms: a fair coin has 1 bit of entropy associated with its unknown outcome (heads or tails). A weighted coin that always comes up heads reveals no additional information when its outcome is observed.

## What is the Shannon Entropy? A Lottery Ticket Example

A coin flip may be low stakes (or not)

no country for old men image

We all might agree that predicting a winning lottery ticket would be tremendously valuable. Suppose a million unique lottery tickets are sold, and each has an equal probability of winning. If you know the winning ticket before the drawing occurs, how many more bits of information do you have than a passerby?

## Interactive Code
Change the code to adjust $p$ and see how entropy changes.

:::{code-cell} python
:tags: [thebe]

import numpy as np
import matplotlib.pyplot as plt
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
:::

Experiment: change p toward 0 or 1: what happens to Shannon entropy?

## Relevance to Deep Learning
Loss function discussion -- cross entropy, KL divergence. Minimize the amount of additional information you need to predict the outcome

## Advanced Applications: Kolmogorov-Sinai Entropy in Training Dynamics

Building on Shannon entropy, Kolmogorov-Sinai (KS) entropy asks a similar question. Consider a particle moving probabilistically and in discrete time. It has many possible surrounding hypercubes it could end up in at time $t+1$. In this sense, what is the entropy of the particle?

We hope to show that:
1. The loss landscape is what informs the prediction probabilities. Therefore, what we've learned about second-order optimizers can be combined with this research area.
2. Via the second law of thermodynamics, the amount of information we learn during a single batch during SGD has the upper bound of the change in KS Entropy; multiple batches have a component
3. IBM research suggests the highest-noise direction on the loss landscape from batch to batch is the principal singular direction
4. Muon optimizer has revolutionized training efficiency by descending under the spectral norm, decreasing the principal eigenvector.
3. It suggests further research be done on effiiciency in learning

### KS Entropy: From Shannon to Trajectories
KS entropy is similar to Shannon entropy but applied over time-evolving states. Consider space partitioned into hypercubes (cells). A trajectory is a sequence of visited hypercubes. KS entropy asks: "Given the current hypercube, how many bits of information do we need to predict where the particle goes next?" It’s the supremum over partitions ξ of the entropy rate:

:::{math}
:name: eq:ks_entropy

h_{KS} = \sup_{\xi} \lim_{n \to \infty} \frac{1}{n} H(\xi^n)
:::

Where \( H(\xi^n) \) is the Shannon entropy of the joint partition after n steps, measuring how much new information is generated per step.

### Visualization: Flat Field vs. Sharp Well
To build intuition, compare two scenarios in a simplified 2D phase space (grid of hypercubes):
- **Flat Field**: No potential gradient—equal probabilities to neighboring hypercubes. High KS entropy (maximal uncertainty in next step).
- **Sharp Well**: Strong potential pull (e.g., toward a minimum)—fewer likely directions. Low KS entropy (predictable trajectory).

Use the interactive tool below: Toggle scenarios or adjust "sharpness" (bias strength). The bar shows approximate KS entropy (computed as Shannon entropy over next-step probabilities, approximating the rate for this partition).

:::{code-cell} python
:tags: [thebe]

import numpy as np
import matplotlib.pyplot as plt

def ks_entropy(probs):
    probs = np.array(probs)
    probs = probs[probs > 0]  # Avoid log(0)
    return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0.0

def plot_ks_scenario(scenario='Flat Field', sharpness=0.8):
    # Simplified 3x3 grid; center is current position (hypercube 4)
    directions = ['NW', 'N', 'NE', 'W', 'Stay', 'E', 'SW', 'S', 'SE']
    if scenario == 'Flat Field':
        probs = [1/8] * 8 + [0]  # Equal to 8 directions, no stay
    else:  # Sharp Well: Bias toward South (e.g., down the well)
        bias = sharpness  # 0: flat, 1: fully biased to one direction
        probs = [(1 - bias)/8] * 8 + [0]
        probs[7] += bias  # Bias to 'S'
    
    entropy = ks_entropy(probs)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(directions, probs, color='lightgreen')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title(f'{scenario}: KS Entropy ≈ {entropy:.2f} bits/step')
    plt.xticks(rotation=45)
    plt.show()
:::

Observe: In the flat field, uniform probabilities yield high entropy (~3 bits, log2(8)). In the sharp well, increasing sharpness concentrates probability, reducing entropy toward 0 (deterministic path).

### Relating KS Entropy to Training Dynamics
In deep learning optimization, parameters evolve as a trajectory in high-dimensional space. KS entropy captures the "chaos" of this process:
- **High KS Entropy (Flat Landscapes)**: Early training or wide minima—updates are exploratory and unpredictable, like a particle in a flat field. This can lead to better exploration but risks instability (e.g., exploding gradients, per syllabus). In chaotic recurrent NNs, h_{KS} equals the sum of positive Lyapunov exponents, linking to sensitive dependence.
- **Low KS Entropy (Sharp Wells)**: Convergent phases or narrow minima—trajectories are constrained, with predictable descent. This promotes stability but may overfit if too rigid. Scaling laws suggest optimal h_{KS} balances exploration and convergence for large models.