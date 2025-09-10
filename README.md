# MAPSS: Source Separation Evaluation Framework

Evaluates source separation algorithms using the Perceptual Separation (PS) and Perceptual Match (PM) measures.

## Installation

```bash
pip install -r requirements.txt

from speech_eval import run_experiment

manifest = [{
    "mixture_id": "mix1",
    "references": [ref1_path, ref2_path],
    "systems": {
        "algorithm": [out1_path, out2_path]
    }
}]

run_experiment(models=["wav2vec2"], mixtures=manifest)
