"""Main entry point."""
from pathlib import Path
from engine import run_experiment

MANIFEST = [
    {
        "mixture_id": "female_pair_example",
        "references": [
            Path(r"C:\postdoc\R1 - PS and PM\Data - speech separation MOS\Unified\Signals\orig\female_inst_sim_1.wav"),
            Path(r"C:\postdoc\R1 - PS and PM\Data - speech separation MOS\Unified\Signals\orig\female_inst_sim_2.wav"),
        ],
        "systems": {
            "Algo1_SASSEC": [
                Path(r"C:\postdoc\R1 - PS and PM\Data - speech separation MOS\Unified\Signals\Algo1_SASSEC\female_inst_sim_1.wav"),
                Path(r"C:\postdoc\R1 - PS and PM\Data - speech separation MOS\Unified\Signals\Algo1_SASSEC\female_inst_sim_2.wav"),
            ],
        }
    },
    {
        "mixture_id": "male_pair_example",
        "references": [
            Path(r"C:\postdoc\R1 - PS and PM\Data - speech separation MOS\Unified\Signals\orig\male_inst_sim_1.wav"),
            Path(r"C:\postdoc\R1 - PS and PM\Data - speech separation MOS\Unified\Signals\orig\male_inst_sim_2.wav"),
        ],
        "systems": {
            "Algo1_SASSEC": [
                Path(r"C:\postdoc\R1 - PS and PM\Data - speech separation MOS\Unified\Signals\Algo1_SASSEC\male_inst_sim_1.wav"),
                Path(r"C:\postdoc\R1 - PS and PM\Data - speech separation MOS\Unified\Signals\Algo1_SASSEC\male_inst_sim_2.wav"),
            ],
        }
    },
]


if __name__ == "__main__":
    results = run_experiment(
        models=["wav2vec2"],
        mixtures=MANIFEST,
        verbose=True,
        max_gpus=None  # Use all available
    )

    print(f"Results saved to: {results}")