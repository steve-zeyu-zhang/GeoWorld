# GeoWorld method and implementation

GeoWorld combines Hyperbolic JEPA (H-JEPA), Geometric Reinforcement Learning
(GRL), and energy-based planning. This document describes the latent geometry,
training objectives, and discrete action search.

## Poincare latent geometry

The latent manifold is the Poincare ball with sectional curvature `-c`, where
`c > 0` is a learned scalar. A frozen visual encoder produces Euclidean features.
`predictor.encode_states(features)` applies the configured fixed tangent scale
and then maps features into the ball:

```text
exp_0(v) = tanh(sqrt(c) ||v||) v / (sqrt(c) ||v||)
log_0(x) = atanh(sqrt(c) ||x||) x / (sqrt(c) ||x||).
```

The limits at zero are the identity map. Curvature is optimized through
`log(c)`, initialized at `c=1`, and clamped to `[0.1, 10]`.
The default fixed tangent scale is `0.01`. Scaling
helps avoid saturation near the ball boundary for large encoder feature norms.
The same encoder, scaling, and curvature must be used for training states,
planning start states, and planning goals.

The metric is

```text
d_c(x,y) = acosh(1 + 2c ||x-y||^2 /
                  ((1-c||x||^2)(1-c||y||^2))) / sqrt(c).
```

`geoworld.geometry.distance` evaluates its equivalent form

```text
d_c(x,y) = 2 asinh(sqrt(c) ||x-y|| /
                  sqrt((1-c||x||^2)(1-c||y||^2))) / sqrt(c).
```

This avoids cancellation and the singular derivative of `acosh` at argument
1. Identical points have distance exactly zero with a zero subgradient.
Curvature remains in the differentiation graph through all maps and distances.
This metric gives `d_c(0, exp_0(v)) = 2 ||v||` for points
unaffected by numerical boundary projection.

Floating-point representations use a radial boundary margin: `1e-5` in FP32
and `1e-10` in FP64, measured relative to radius `1/sqrt(c)`. Projection leaves
interior points unchanged. Geometry requires FP32 or FP64; attention precision
can be configured separately. Extremely large tangent vectors saturate at this
numerical boundary, so invertibility is only tested below saturation.

## Supervised post-training

Training input consists of encoder features `[B,T+1,D]`, fixed-vocabulary
action labels `[B,T]`, and a Boolean valid-transition mask `[B,T]`.
The encoder remains frozen. A causal action-conditioned predictor operates on
hyperbolic latent states.

Teacher forcing (Equations 4–6) feeds the encoded state sequence and
actions to the predictor and compares each predicted next state with the next
encoded state using hyperbolic distance. Sequence attention is causal, so a
prediction cannot observe later states or labels.

Two-step supervision (Equations 7–9) uses every adjacent action pair:
start from its encoded state and recursively predict twice. The second step
uses the first prediction, retaining the generated causal history, and is
compared with the encoded state two transitions later.

```text
L_SFT = lambda_tf * L_teacher + (1-lambda_tf) * L_two_step.
```

The default `lambda_tf=0.5` gives equal weight to both terms
(Appendix Table 3). Each term averages valid windows within
each trajectory, then averages trajectories with at least one such window.
An example with no two-step window contributes no sample to that term. If the
entire batch has no two-step window, its rollout term is zero and the teacher
term retains its configured weight. An entirely empty transition batch is
rejected. Padding labels may be `-1`; valid labels must be nonnegative.

## Geometric Reinforcement Learning

GRL recursively predicts a complete trajectory from its first encoded state.
Equation 18 minimizes discounted hyperbolic prediction energy:

```text
energy = mean_batch sum_t gamma^t d_c(predicted_state[t+1], target_state[t+1])
L_GRL = energy + beta * L_triangle.
```

The defaults are `gamma=0.99` and `beta=0.1` (Section 4.3 and Appendix Table 4).
The horizon reduction is a **sum**, not a temporal mean. Padded transitions
are omitted, and valid transitions must form a contiguous prefix. The reported
return is `-energy`, consistent with the negative-energy reward in Equation 13.
Maximizing that return and minimizing energy have the same optimizer; their
optimal numerical values differ by a sign.

GRL uses observed dataset action trajectories. Gradients propagate through
the world-model rollout to optimize the predictor and curvature. Training
uses no separate policy network, reward model, or policy-gradient objective.

### Triangle regularization

Equation 17 defines the regularizer over consecutive predicted states:

```text
L_triangle = mean [d_c(s_t,s_t+2) - d_c(s_t,s_t+1) - d_c(s_t+1,s_t+2)]_+.
```

For an exact metric, the triangle inequality guarantees that the expression
inside the hinge is nonpositive for every triplet. **The regularizer is
identically zero in exact arithmetic**, including for paths that are not
geodesics. Changing `beta` therefore cannot change the objective or its
gradient in exact arithmetic. Floating-point error can produce tiny positive
values. Tests include a non-geodesic curve whose triangle hinge remains zero.

## Energy-based planning in the discrete action space

The Cross-Entropy Method (CEM) searches sequences of labels from the dataset's
fixed action vocabulary. The default action space is the **entire dataset
vocabulary**. Initial distributions are uniform and factorized by planning
step; no additional action proposal or policy model is trained.

For each refinement iteration:

1. Sample `N` complete action sequences from the categorical distributions.
2. Roll out the frozen predictor from the current state for every candidate.
3. Score only the final predicted state against the encoded goal using
   hyperbolic distance, as in Equation 20.
4. Select the lowest-energy `K` sequences and refit each step's categorical
   distribution from the elite label frequencies.

The defaults are `N=800`, `K=80`, and `I=10` refinement iterations.
The best sequence found so far is retained in the next candidate population.
An exploration mixture of `0.05` with the allowed uniform distribution avoids
prematurely assigning zero probability to a label; this is a configurable
numerical search setting. Candidate evaluation is chunked to limit memory.
The returned plan is the best complete sequence, together with its final
energy, predicted trajectory, categorical probabilities, and energy history.

An optional Boolean action mask can have shape `[A]`, `[B,A]`, or `[B,T,A]`.
Masked labels are never sampled and retain zero probability. At least one
action must remain allowed at every step. Restricting labels using task identity
is an additional planning condition and must be reported in evaluation; it is
disabled by default. Goal scoring uses no ground-truth intermediate states.

Planning runs under `no_grad`; the predictor should be in evaluation mode.
An explicit `torch.Generator` controls candidate sampling. The same seed,
inputs, and deterministic predictor yield the same candidate search.

## Verification and scope

Geometry tests compare the stable distance with its `acosh` form, verify
Exp/Log inversion and radial distances, check learned-curvature gradients, and
exercise zero vectors and the numerical boundary. Loss tests use known
trajectories to verify teacher forcing, recursive two-step prediction,
discounting, masks, gradients, and the literal triangle hinge. Planning tests
recover known ordered discrete plans, verify fixed-label and optional masked
search spaces, and check repeatability.

These checks establish mathematical and software behavior. They do not
establish CrossTask/COIN benchmark accuracy, which requires the matching visual
encoder, official data splits and protocol, full training schedules, and
evaluation of the resulting action sequences.
