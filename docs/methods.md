# Experimental methods

Ruliology Forge compares an unperturbed control trajectory with a perturbed trajectory generated from the same initial state.

## Perturbation timing

The default is `after_update`: the automaton first computes row `t`, then the perturbation is applied directly to that row. Thus, control and perturbed trajectories are identical before `perturb_time`, and the visible injury begins exactly at `perturb_time`.

The optional `before_update` mode perturbs row `t - 1` immediately before computing row `t`. Every output records the timing convention.

## Randomness

A root seed is split into independent initial-condition and perturbation seeds with NumPy `SeedSequence`. This allows the two stochastic processes to be reproduced independently.

## Recovery

Recovery occurs when normalized Hamming divergence remains at or below the selected threshold for the requested number of consecutive steps. Recovery time is relative to the perturbation time.

## Scar metrics

The toolkit reports final scar size, duration, maximum spatial spread, and centroid drift of the differing region. These are descriptive computational measures and should not automatically be interpreted as biological scarring.

## Shift-tolerant restoration

A translated pattern may be structurally restored while failing exact cell-by-cell comparison. Shift-tolerant restoration searches within `[-max_shift, max_shift]` and reports the best alignment at each time step. Both exact and shift-tolerant scores should be reported.

## Interpretation caution

Convergence under a chosen metric is evidence of computational restoration under the stated experimental conditions. It is not, by itself, evidence that a biological system uses the same mechanism or that the automaton biologically regenerates.
