# Acquisition Function Derivations

## Expected Improvement (EI)

Let f : X → ℝ be the unknown property function. Given observations D = {(x_i, y_i)},
we model f with a Gaussian Process:

    f(x) | D ~ N(μ(x), σ²(x))

The **improvement** over the current best observation y* = max{y_i} is:

    I(x) = max(f(x) - y*, 0)

**Expected Improvement** is the expectation of this improvement:

    EI(x) = E[I(x)] = E[max(f(x) - y*, 0)]

For a Gaussian distribution N(μ, σ²):

    EI(x) = (μ(x) - y* - ξ) · Φ(Z) + σ(x) · φ(Z)

    where Z = (μ(x) - y* - ξ) / σ(x)

**Interpretation:**
- When σ(x) → 0: EI → max(μ(x) - y*, 0)   — pure exploitation
- When σ(x) >> 0: the σ·φ(Z) term dominates — exploration bonus
- ξ > 0 shifts the threshold down, encouraging more exploration

## PUDA Hybrid Strategy

    PUDA(x) = α · norm(μ(x)) + β · norm(σ(x)) + γ · div(x, S)

    norm(v) = (v - min(v)) / (max(v) - min(v))   [min-max over pool]

    div(x, S) = min_{s ∈ S} cosine_distance(x, s)

### Diversity via k-center Greedy

The greedy k-center problem: select k points to minimise the maximum
distance from any pool candidate to its nearest selected point.

Greedy approximation (2-approximation):
    1. S = {}
    2. For i = 1 to k:
       a. Score all remaining x by PUDA(x) with current S
       b. x* = argmax PUDA(x)
       c. S = S ∪ {x*}

After each pick, div(x, S) is updated for all remaining x.

**Why cosine distance**: Material descriptor vectors may have wildly
different magnitudes across features. Cosine distance is scale-invariant,
measuring angular similarity, which better reflects compositional
similarity than Euclidean distance.
