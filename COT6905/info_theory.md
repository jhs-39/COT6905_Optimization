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

# Information Theory: Introduction to Shannon Entropy

## Overview
Information theory provides tools to quantify uncertainty and information content of data. It formulates a quantity called 'information' that determines how much a measurement decreases uncertainty, or Shannon Entropy, of underlying data. Tools from information theory form the backbone of neural net loss functions which are used to train models. Cross-Entropy loss and KL Divergence are used extensively for this purpose, which are derived from information-theoretic formulations. Another place where Shannon Entropy is used is to analyze the output of predictive models. The Shannon Entropy can be applied to these outputs to characterize how uncertain the model is about its predictions, critical for evaluating models for risk-aware deployments.

Finally, we ask a question of Shannon Entropy in regards to training dynamics of a large neural net. Can we upper bound the amount of information learned in a single batch of data during stochastic gradient descent?

## Shannon Entropy Formula
For a categorical probability vector $P = [p_1, p_2, \dots, p_n]$ where $\sum p_i = 1$:

$$
H(P) = -\sum_{i=1}^n p_i \log_2 p_i
$$

We use base-2 logarithms to measure in *bits*. For \( p_i = 0 \), the term is defined as 0.

## What is the Shannon Entropy? A Binary Entropy Example
Let's simplify to a binary distribution, aka a coin flip: probabilities \( p \) and \( 1-p \). The entropy is:
$$
H(p) = -p \log_2 p - (1-p) \log_2 (1-p)
$$
This reaches a maximum of 1 bit at \( p = 0.5 \) (complete uncertainty) and 0 at the extremes (certainty). In simple terms: a fair coin has 1 bit of entropy associated with its unknown outcome (heads or tails). A weighted coin that always comes up heads reveals no additional information when its outcome is observed.

## What is the Shannon Entropy? A Lottery Ticket Example

A coin flip may be low stakes (or not)

no country for old men image

We all might agree that predicting a winning lottery ticket would be tremendously valuable. Suppose a million unique lottery tickets are sold, and each has an equal probability of winning. If you know the winning ticket before the drawing occurs, how many more bits of information do you have than a ignorant citizen?

```{code-cell} python
import math

# number of equally likely outcomes
N = 1_000_000

H_bits = math.log2(N)   # Shannon entropy in bits. See that the sum over N terms cancels out with p_i = 1/N coefficient in front of each term
#See that Shannon Entropy of a uniform probability distribution scales with the log number of outcomes!

print(f"N = {N}")
print(f"Entropy: {H_bits:.12f} bits")
```

## OK, sure, but what is the Shannon Entropy?

Why does the Shannon entropy formula involve the logarithm of the probability, specifically $\log_2 p_i$? This arises from the fundamental nature of information in decision trees or binary questioning processes, where each "bit" of information halves the space of remaining possible outcomes. Suppose you hack the lottery ticket system, and discover an exploit. You can predict the last binary digit of the soon-to-be-winning ticket number! You predict '1.' This naturally eliminates all the ticket numbers that end with '0', or exactly half the possibilities.

### Binary Trees and Logarithmic Scaling

Imagine you're trying to identify a specific outcome from a set of equally likely possibilities. One way to gather information is through yes/no questions that split the possibilities in half each time—like a binary search. For (N) equally likely outcomes, the number of questions needed (on average) is $\log_2 N$, because each question eliminates half the possibilities.

For $N = 2$ (e.g., a fair coin flip), $\log_2 2 = 1$ bit: one question suffices ("Heads?").

For $N = 4$, $\log_2 4 = 2$ bits: two questions halve the space twice.

This logarithmic scaling captures the "height" of a balanced binary decision tree. The negative sign in $H(P) = -\sum p_i \log_2 p_i$ ensures entropy is positive (since $p_i < 1$, $\log_2 p_i < 0$), and the sum weights it by each probability. For non-uniform distributions, it generalizes this idea: low-probability events convey more information (larger $-\log_2 p_i$) when they occur, as they eliminate more uncertainty.

### Information Learned as Change in Entropy

Information isn't just about the initial uncertainty—it's about the reduction in entropy after an observation or experiment. If you start with prior entropy $H(P)$ and update to a posterior distribution $Q$ after new data, the information gained is $H(P) - H(Q)$. (This is related to mutual information or Kullback-Leibler divergence, which we'll cover later)

In deep learning, this is crucial: training reduces model uncertainty about data (e.g., via cross-entropy loss, which measures the "extra bits" needed to encode true labels under the model's predictions). A model with high initial entropy (uncertain predictions) learns by minimizing this gap.

### Classic Example: The Ball-Weighing Puzzle (from MacKay)

To illustrate designing experiments that maximize information gain, consider this puzzle from David MacKay's Information Theory, Inference, and Learning Algorithms: You have 12 balls that appear identical, but one is either heavier or lighter than the others (you don't know which or in what direction). You have a balance scale with two sides. How do you identify the odd ball and its defect in the fewest number of weighings?

Initial Entropy: There are 12 balls × 2 defects (heavy/light) = 24 possibilities. Entropy $H = \log_2 24 \approx 4.58$ bits. This is our starting entropy

How much is your change in entropy if you weigh 6 balls vs 6 balls? Hint: this measurement has only two possible outcomes. What is $log_2 2$?

Draw out the tree on paper, and follow the code below. Suppose the left side of the scale drops. This means the defect is either in the left side as a 'light ball' (6 possibilities) or in the right side as a 'heavy ball' (6 possibilities) giving us 12 possibilities from the original 24. What is our new entropy, once we have this measurement?

Can you imagine a split for this measurement with more possible outcomes than two?

That creates a rule for experiments: to maximize information gain, maximize the number of equally weighted possible outcomes!

Run the interactive code below to simulate a single weighing. Adjust group sizes and see how the entropy reduction changes. Can you devise an initial weighing step that has 3 distinct outcomes instead of only 2?

## Interactive Code
Change the code to adjust $p$ and see how entropy changes when you construct a different measurement

```{code-cell} python
:tags: [thebe]
import numpy as np
import matplotlib.pyplot as plt
import math

def entropy_reduction(left_group=6, right_group=6, total_balls=12):
    defects=1
    total_poss = total_balls * defects  # Initial possibilities
    initial_h = math.log2(total_poss)
    
    unbal_left = left_group * defects  # Poss if left heavy/light
    unbal_right = right_group * defects
    bal = (total_balls - left_group - right_group) * defects
    
    # Normalize probs
    p_unbal_l = unbal_left / total_poss
    p_unbal_r = unbal_right / total_poss
    p_bal = bal / total_poss
    
    # Conditional entropies (simplified: assume uniform within branches)
    h_unbal_l = math.log2(unbal_left) if unbal_left > 0 else 0
    h_unbal_r = math.log2(unbal_right) if unbal_right > 0 else 0
    h_bal = math.log2(bal) if bal > 0 else 0
    
    # Expected posterior entropy
    exp_h_post = p_unbal_l * h_unbal_l + p_unbal_r * h_unbal_r + p_bal * h_bal
    
    info_gain = initial_h - exp_h_post
    
    # Plot
    labels = ['Initial H', 'Expected Post H', 'Info Gain']
    values = [initial_h, exp_h_post, info_gain]
    colors = ['gray', 'lightgray', 'skyblue']
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=colors)
    ax.set_ylabel('Bits')
    ax.set_title(f'Entropy Reduction for Weighing {left_group} vs. {right_group}')
    plt.show()

# Example: Change left/right group sizes!
entropy_reduction(left_group=6, right_group=6)

```
(if you didn't get it, try 4 and 4 for your measurement. The balls can either be weighted to the left, right, or the scales can be balanced. This gives us a change in entropy of $log_2 3$, or equivalently it eliminates 16 possible states from the original 24, leaving only 8. Likewise, calculate the change in entropy and compare it to number of possibilities left.)

## Advanced Applications: Kolmogorov-Sinai Entropy in Training Dynamics

Building on Shannon entropy, Kolmogorov-Sinai (KS) entropy asks a similar question. Consider a particle moving probabilistically and in discrete time. It has many possible surrounding hypercubes it could end up in at time $t+1$. In this sense, what is the entropy of the particle?

We hope to show that:
1. The loss landscape is what informs the prior probabilities that bound the amount of information transferred. Therefore, what we've learned about second-order optimizers can be combined with this research area.
2. Via the second law of thermodynamics, the amount of information we learn during a single batch during SGD has the upper bound of the change in KS Entropy. Given that gradients are notoriously low-rank (IBM Research), the actual degrees of freedom are quite low, on the order of log(params)
3. IBM research suggests the highest-noise direction on the loss landscape from batch to batch is the principal singular direction
4. Muon optimizer has revolutionized training efficiency by descending under the spectral norm, decreasing the principal eigenvector.
5. It suggests further research be done on effiiciency in learning. What methods maximize information transfer?

## Math for KS Entropy in Training Dynamics

We guide this discussion with the 2nd law of thermodynamics; in a closed system, the change in entropy must be greater than or equal to 0. In its extension to information theory and learning, in the closed system described by the model and data batch, the total change in entropy must be greater than or equal to 0. This means that the upper bound of 'useful' bits learned by the model must be given by the entropy generated by the SGD batch.

Suppose I have $n$ parameters in a neural net, and initialize at a point $\theta_0$ in the parameter space. After SGD at learning rate R over batch 1, the model has moved to point $\theta_1$ in parameter space. Can we express the number of possible states of $\theta_1$ as the surface area of the high dimensional shell, divided by the spherical cap area given by quantization? Under the (bizarre) case that all model variables were independent, this would be the upper bound of the information transferred.

Area of the d-Dimensional Hypersphere
$$
A = \frac{2 pi ^ d/2 r^d-1}{\gamma(d/2)} 
$$

Speherical Cap given by quantum epsilon
$$
\sin{\epsilon}^{d-2} \epsilon \approx \epsilon{d-1}
$$

Number of possible microstates in d dimensions with quantum epsilon
$$
log_2{\frac{A}{\epsilon^{n-1}}
$$

Plugging in numbers for a 1-billion-parameter model with quantum 

Yikes! That's a large number. Imagine transferring gigabytes of information per update!

However, in real life, gradient updates are much lower rank; the parameters are far, far from independence. IBM's work shows that 99% of the variance in a layer's gradients is captured by a rank-40 matrix. In that case, we see: 

Several hundred bits. A much more reasonable upper bound.

Finally, let's consider temperature. IBM's definition of temperature is actually anisotropic; the noise on the loss landscape actually varies with the direction you consider. There are high-noise directions that greatly vary from batch to batch (primarily the direction of the principal singular value), and low-noise directions that are the same batch-to-batch.

We can consider a Helmholtz free energy equivalent to see how temperature affects learning. 
F = U - TS

Recall Helmholtz free energy describes the amount of energy available to do work. In our case, we are drawing the parallel to information available to do useful work. If we learn a great deal about a particular batch (and overfit) 