# Overview
This library implements the computational framework introduced in "Bi-Conflicts and Qualitative Dominance in Hierarchical Conflict Analysis" (forthcoming). It provides tools for enumerating and filtering coalition structures, opposition structures, and bi-conflicts from a multi-agent attitude matrix, using Pareto-optimal dominance to identify the most significant conflicts within hierarchically prioritised groups of agents and issues.

The usage of the library is presented using popular Middle east conflict situation. The details are described within each file:
- pareto_conf_general.py is the source code of the library
- middle_east_experiments.ipynb presents how to use this library

## Implementation Notes

The model stores agent and issue importance classes alongside a `{-1, 0, 1}` attitude matrix. Each coalition or opposition structure is augmented with integer rank encodings of its agent and issue profiles — mixed-radix numbers that preserve lexicographic profile order — so that dominance comparisons reduce to simple integer comparisons.

All Pareto frontiers avoid the naïve O(n²) pairwise scan. The **Q-frontier** is computed as a 2D skyline on the two rank values in O(n log n) via sort-and-sweep. The **Q\*-frontier** first applies the Q-skyline as a pre-filter (anything eliminated there cannot survive the stricter relation), then runs the full pairwise check only within the small surviving set, costing O(n log n + f²) where f ≪ n in practice. **Bi-conflict frontiers** decompose the problem by computing structure-level frontiers over coalition and opposition components independently, using these to prune candidates, and finishing with a pairwise check on the small remainder — the same two-phase strategy applied to both the Q and Q\* variants.
