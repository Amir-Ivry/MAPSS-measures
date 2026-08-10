# Using MAPSS in a grand challenge

## Recommended organizer integration

Pin one released version in the challenge evaluator:

```text
mapss-measures==1.1.1
```

For each mixture, pass the ordered reference sources and the corresponding ordered system
outputs to `mapss`. Store the frame-level CSVs, not only the displayed means.

```python
from mapss import mapss

scores = mapss(
    reference=reference_paths,
    output=submission_paths,
    source_names=speaker_ids,
    model="wav2vec2",
    layer=2,
    alpha=1.0,
    seed=42,
    max_gpus=1,
)
scores.save(output_directory)
```

The evaluator should validate the source-to-reference assignment before calling MAPSS. MAPSS
deliberately does not guess or optimize that assignment because silent permutation changes would
make systems incomparable.

## Recommended public challenge statement

> We report MAPSS Perceptual Separation (PS) and Perceptual Match (PM) using
> `mapss-measures==1.1.1`, wav2vec 2.0 Large layer 2, `alpha=1`, and seed 42.
> Higher is better. Outputs are ordered to match their reference sources. Frames with fewer
> than two active references are excluded. Code and definitions are available in the MAPSS
> repository and ICLR 2026 paper.

Link the words “MAPSS repository” to
`https://github.com/Amir-Ivry/MAPSS-measures` and “ICLR 2026 paper” to
`https://arxiv.org/abs/2509.09212`.

## Aggregation policy

The package returns frame-level values. Freeze one aggregation policy before submissions are
evaluated and apply it identically to every system.

- PM: the paper averages active-frame values for an utterance.
- PS: the paper's formal utterance aggregation is defined in Appendix B.4 and is not simply a mean.
- Cross-item leaderboard aggregation: state whether items, speakers, or duration receive equal weight.

Do not convert inactive `NaN` frames to zero. That would penalize silence rather than separation.

## Reproducibility checklist

- Pin the package version and dependency lock or container image.
- Freeze model, layer, alpha, seed, CI setting, and aggregation.
- Preserve source order and source identifiers.
- Run one public toy example and publish its expected score files.
- Evaluate all teams on the same hardware class when runtime is part of the protocol.
- Archive the exact evaluator commit and cite the paper.

## Practitioner-facing announcement

> MAPSS is now available as a Python package for frame-level evaluation of source
> separation. Give it ordered reference and output waveforms; it returns complementary
> Perceptual Separation (leakage) and Perceptual Match (self-distortion) scores in `[0, 1]`,
> with optional paper-derived error quantities. Install version 1.1.1 with
> `pip install mapss-measures==1.1.1`, view the quick start, and cite the ICLR 2026 paper.
