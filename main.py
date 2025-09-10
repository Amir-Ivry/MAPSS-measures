"""Main entry point."""

from pathlib import Path

from engine import run_experiment

SASSEC_PATH = "/disks/disk1/scornell/mapss/MAPSS-measures/SASSEC"  # full path to SASSEC root folder, do not use relative path


MANIFEST = [
    {
        "mixture_id": "female_pair_example",
        "references": [
            Path(SASSEC_PATH) / Path("Signals/orig/female_inst_sim_1.wav"),
            Path(SASSEC_PATH) / Path("Signals/orig/female_inst_sim_2.wav"),
        ],
        "systems": {
            "Algo1_SASSEC": [
                Path(SASSEC_PATH) / Path("Signals/Algo1/female_inst_sim_1.wav"),
                Path(SASSEC_PATH) / Path("Signals/Algo1/female_inst_sim_2.wav"),
            ],
        },
    },
    {
        "mixture_id": "male_pair_example",
        "references": [
            Path(SASSEC_PATH) / Path("Signals/orig/male_inst_sim_1.wav"),
            Path(SASSEC_PATH) / Path("Signals/orig/male_inst_sim_2.wav"),
        ],
        "systems": {
            "Algo1_SASSEC": [
                Path(SASSEC_PATH) / Path("Signals/Algo1/male_inst_sim_1.wav"),
                Path(SASSEC_PATH) / Path("Signals/Algo1/male_inst_sim_2.wav"),
            ],
        },
    },
]


if __name__ == "__main__":
    results = run_experiment(
        models=["wav2vec2"],
        mixtures=MANIFEST,
        verbose=True,
        max_gpus=None,  # Use all available
    )

    print(f"Results saved to: {results}")
