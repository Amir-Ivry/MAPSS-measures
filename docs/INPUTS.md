# Input contract and guardrails

MAPSS evaluates one mixture containing `N >= 2` aligned source pairs. The two ordered
collections are the primary contract:

```text
reference[0] <-> output[0]
reference[1] <-> output[1]
...
reference[N-1] <-> output[N-1]
```

MAPSS does not infer or change this assignment. A count mismatch reports both received
counts and stops before model loading.

## Accepted inputs

`reference` and `output` may each be:

- a list or tuple of audio paths;
- a list or tuple of one-dimensional NumPy arrays or PyTorch tensors;
- a two-dimensional NumPy array or PyTorch tensor shaped `(sources, samples)`;
- a mixture of path-like objects within the same ordered collection.

For an individual multi-channel in-memory waveform, use `(samples, channels)`. MAPSS
rejects likely `(channels, samples)` input with transpose guidance. File inputs are decoded
by SoundFile/libsndfile. WAV and FLAC are covered by end-to-end tests; stereo and different
source sample rates are also tested. Other formats work only when the installed libsndfile
build can decode them. Corrupt or unsupported files produce an error naming the exact input
and path.

## Conversion and duration

Every source is independently downmixed to mono and resampled to 16 kHz. MAPSS then
requires all converted references and outputs to have exactly equal duration. The default
`length_policy="error"` reports the duration and sample count of every input when they
differ.

Use `length_policy="trim"` only when discarding all samples beyond the shortest input is
scientifically intended. It never pads or time-aligns signals. Every input must remain at
least 400 ms after conversion/trimming.

## Rejected before expensive processing

MAPSS raises a specific error for:

- fewer than two source pairs;
- more references than outputs or more outputs than references;
- missing files, directory paths, corrupt audio, or unsupported encodings;
- empty, silent, nonnumeric, NaN, or infinite waveforms;
- unsupported waveform dimensions or likely channel-first arrays;
- unequal converted durations under the default policy;
- missing, duplicate, blank, or reserved source names;
- invalid model, layer, alpha, seed, GPU, or Boolean options;
- explicit CUDA requests when PyTorch cannot access a CUDA GPU;
- audio too short for the 400 ms loudness-analysis block;
- any source with no valid PS or PM frame in which two or more references are
  simultaneously active;
- malformed score/confidence tables during plotting.

## Formats versus extensions

A filename extension does not guarantee that a file contains valid audio. MAPSS attempts
to decode the content and reports a decoding error if it fails. Converting unusual challenge
formats to lossless WAV or FLAC before evaluation is the most portable option. Do not use
lossy transcoding merely to satisfy an extension requirement.

## Tested source counts

The release integration suite runs the complete raw-waveform engine for `N=2`, `N=3`, and
`N=4`, including confidence calculation and paper-style plotting. The public API is not
hard-coded to four sources, but larger mixtures require more memory and computation and
should be validated on representative challenge data.
