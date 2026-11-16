# Origin and adaptation

This standalone research harness is adapted from the generic baseline research core of Aleksandar Kamenev's personal ATLAS lab. ATLAS is an independent, AI-assisted prototype. This adaptation was prepared in November 2026.

## Existing core

The scoped adaptation preserves the original generic algorithms for BaseRate, Oracle, ItemOnly, BKT, EloIRT and PFA, along with synthetic learner generation, replay and metrics. Their presence here is an engineering adaptation of established model families, not a claim to have invented those algorithms. Primary lineage references are listed in the [README](README.md); the implementation variants and evaluation boundaries are recorded in [Implementation notes](IMPLEMENTATION.md).

## New release work

AI assistance was used during this task to isolate the core, prepare new neutral synthetic fixtures and write the standalone package/CLI and documentation. The fixture surfaces are pattern, sequence and allocation. These new release components should not be presented as pre-existing deployment history.

## Boundary

The standalone scope excludes the game's content, assets, story, game-specific mastery/pacing policies, original game-derived item bank, conformance vectors and game exports. Neural/GPU features are also excluded. It contains synthetic experiments rather than real learner records and provides no policy export for the game.

This harness is a prototype research tool, not peer-reviewed evidence or a human learning-effectiveness study. Results depend on the simulation assumptions. Fresh verification of the standalone adaptation must be recorded separately from any earlier checks of the personal lab.

The scoped standalone release is licensed under [MIT](LICENSE), copyright 2026 Aleksandar Kamenev.
