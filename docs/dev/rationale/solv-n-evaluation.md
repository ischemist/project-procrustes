---
icon: lucide/clipboard-check
---

# Solv-N Evaluation

This page explains the implementation of the Solv-N evaluation framework in RetroCast.

## Historical Context

RetroCast originally used "solvability" in in the common retrosynthesis sense: a route was solved if it terminated in the selected stock.

- The original RetroCast preprint [arxiv:2512.07079](https://arxiv.org/abs/2512.07079) highlighted that such "stock-termination rate" is an insufficient measure of success.
- and the following [Syntax of Matter preprint](https://ischemist.com/syntax-of-matter) formalized the Tier-N hierarchy of chemical validity and proposed the Solv-N metric system.

## Core Separation

Fundamentally, Solv-N combines two key concepts:

- all reactions composing a `Route` must be "valid" at some level of validity (internal, chemical constraints)
- the `Route` must solve the problem task (user-defined constraints)

```text
Solv-i[task] = Tier-i validity + satisfying task constraints
```

The bracketed metric label should name the task semantics, not every per-target value. For example, stock termination against a purchasable set is `Solv-i[buyables]`; adding target-specific route-depth constraints gives `Solv-i[buyables+depth]`; adding target-specific required leaves gives `Solv-i[buyables+leaf]`. Exact constraint values remain in the task artifact.

### Tier-N Chemical Validity

As a refresher (consult [Syntax of Matter](https://ischemist.com/syntax-of-matter) for more details):

- Tier 0 validity ensures all proposed SMILES correspond to valid chemical structures (e.g. satisfying basic valency rules)
- Tier 1 validity ensures all reactions are topologically valid (e.g., you can extract a valid SMARTS template)
- Tier 2 validity ensures all reactions satisfy chemoselectivity, regioselectivity, diastereoselectivity, enantioselectivity, and stoichiometry
- Tier 3 validity ensures all reactions are experimentally viable

If a `Route` is Tier-2 valid, it is experimentally _plausible_. If a `Route` is Tier-3 valid, it is experimentally _feasible_. Currently, we lack a systematized and universal way to assess even Tier-2 validity, so RetroCast is built to:

1. Assess Tier-0 and Tier-1 validity
2. be modular enough to incorporate any external Tier-2 validity check

### Problem Task Satisfaction

Tier-N validity is not enough. A `Route` might start with a target, which will undergo Tier-3 valid reactions, but it still might not be a solution to the retrosynthesis problem if this `Route` does not terminate in commercially available building blocks. Satisfaction of problem constraints is what turns Tier-N validity into a Solv-N metric.

This definition allows for clear generalization of Solv-N to other problems:

- in synthesis-aware molecular design (forward planning from a set of commercial building blocks), task satisfaction is "forward plan terminates in desired target" or "the target satisfies desired properties"
- in constrained versions of retrosynthesis, i.e. bidirectional planning, task satisfaction is "the Route terminates in the commercial stock AND one of the leaves is whatever the user specified"

### Mean-Reverse Rank (MRR) is a companion to Solv-N

Solv-N measures if a model finds any `Route` that satisfies the problem constraints and all its reactions are Tier-N valid. A user of the planner might also be interested in whether the model prioritizes the Tier-N valid `Routes` or he has to go through 50 predictions before finding a valid one.

This is measured by mean-reverse rank (MRR@Solv-N) metric.

## Acceptable Route Reconstruction

In the absence of automated Tier-2 validity checks, as a temporary proxy of full chemical validity we test whether a model can reconstruct an existing, experimentally-verified route for a novel target.

Top-K accuracy measures if an experimentally verified `Route` is reconstructed within first `k` `Routes` returned by the model. A candidate reconstructs an acceptable route when the acceptable route is a target-rooted prefix of the candidate route: `candidate.signature(depth=acceptable.depth()) == acceptable.signature()`, with the candidate at least as deep as the acceptable route. This avoids penalizing a planner that recovers the reference route and then continues expanding below a reference leaf.

Exact full-route identity remains available at scoring time through `acceptable_route_match="exact"` / `--acceptable-route-match exact`, but prefix matching is the default reported reconstruction semantics.

As proposed in the original [RetroCast preprint](https://arxiv.org/abs/2512.07079), we utilize a user-centric evaluation approach. As such, the ranking is performed **after** Routes are filtered for satisfaction of the problem scope (because Routes that do not satisfy them are of no interest to the user).

While Top-K accuracy fails to reward construction of potentially valid alternative route (a limitation well discussed in the Syntax of Matter preprint), the acceptable-route is one of the valid `Routes` that any planner (even the future Tier-3 compliant ones) should consider, and so it is reasonable to expect its reconstruction for some value of K. The exact value of K is up to debate (how many unique ways are there to make any random molecule?), and we think `K=10` and `K=50` are worth paying attention to.

Notably, this means that `Top-1` accuracy should not be the headline metric.

`analyze` also reports reconstruction diagnostics that preserve the same task-satisfaction filter and top-k window:

- `acceptable_root_reconstruction_top_k[...]`: fraction of targets where a useful top-k candidate has the same root reaction as any acceptable route.
- `acceptable_reconstruction_given_root_top_k[...]`: acceptable-route reconstruction rate among targets with a root hit. Its denominator is targets, not candidate routes.
- `acceptable_prefix_reconstruction_depth_d_top_k[...]`: fraction of targets where a useful top-k candidate matches an acceptable route prefix by `Route.signature(depth=d)`.
- `distinct_root_reactions_top_k[...]`: mean number of distinct root reaction signatures among useful top-k candidates.
