# Interpreting MAPSS results

MAPSS provides complementary measures for each assigned source. Values are frame-level and
lie in `[0, 1]`; higher is better.

## Perceptual Separation (PS)

PS measures how well an estimated source is separated from the competing reference
sources. A high PS indicates little perceptually relevant interference from non-target
sources. PS should be interpreted together with PM: an output can suppress interference
while also damaging or omitting its target.

## Perceptual Match (PM)

PM measures how closely an estimated source matches its assigned reference. A high PM
indicates good target preservation under the MAPSS perceptual representation. PM alone does
not describe leakage from other sources, which is why MAPSS reports PS separately.

## Frame tables

`ps_scores.csv` and `pm_scores.csv` contain:

- `timestamp_ms`: the frame time in milliseconds;
- one score column for every source name.

Frames at which the measure is not defined are represented by `NaN`. Do not replace those
values with zero: doing so would convert inactivity into an artificial performance penalty.

## Summary table

`summary.csv` provides a convenient diagnostic summary:

- `ps` and `pm`: arithmetic means over finite, active frames;
- `ps_frames` and `pm_frames`: the number of frames used in each mean.

The summary is useful for inspection, but it is not a substitute for a declared evaluation
protocol. In particular, the paper's formal utterance-level PS analysis uses the pooling
procedure in Appendix B.4 rather than a plain mean. The paper's PM utterance score averages
active-frame values.

## Confidence/error output

When `add_ci=True`, `confidence.csv` contains the deterministic and probabilistic error
components derived by MAPSS. These columns should be interpreted using the definitions and
assumptions in the ICLR 2026 paper; they should not be relabeled as generic confidence
intervals without preserving that definition.

For each source, the four columns are:

- `<source>_pm_bias`: PM deterministic truncation-error radius;
- `<source>_ps_bias`: PS deterministic truncation-error radius;
- `<source>_pm_prob`: PM high-probability 95% error bound;
- `<source>_ps_prob`: PS high-probability 95% error bound.

These values are nonnegative error magnitudes. They are not additional quality scores, and
higher is not better. `result.save(..., plot=True)` places them below PM and PS in the same
order as the paper's Figure 11. The plotted curves show the error magnitudes themselves;
they are not silently converted into clipped lower/upper score envelopes.

## Comparing systems

For a fair comparison:

1. keep the reference/output assignment fixed;
2. use the same package version and hyperparameters;
3. use the same activity and aggregation policy;
4. retain frame-level tables, not only summary means;
5. state how scores are aggregated across mixtures, speakers, or durations.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and
[CHALLENGE_GUIDE.md](CHALLENGE_GUIDE.md) before reporting results.
