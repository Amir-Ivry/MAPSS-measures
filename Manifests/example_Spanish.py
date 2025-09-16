from pathlib import Path

SASSEC_PATH = "C:/postdoc/R1 - PS and PM/Data - speech separation MOS/SASSEC"  # full path

MANIFEST = [
    {
        "mixture_id": "female_pair_example",
        "references": [
            Path(SASSEC_PATH) / Path("Signals/orig/female_inst_sim_3.wav"),
            Path(SASSEC_PATH) / Path("Signals/orig/female_inst_sim_4.wav"),
        ],
        "systems": {
            "Algo1_SASSEC": [
                Path(SASSEC_PATH) / Path("Signals/Algo1/female_inst_sim_3.wav"),
                Path(SASSEC_PATH) / Path("Signals/Algo1/female_inst_sim_4.wav"),
            ],
        },
    },
    {
        "mixture_id": "male_pair_example",
        "references": [
            Path(SASSEC_PATH) / Path("Signals/orig/male_inst_sim_3.wav"),
            Path(SASSEC_PATH) / Path("Signals/orig/male_inst_sim_4.wav"),
        ],
        "systems": {
            "Algo1_SASSEC": [
                Path(SASSEC_PATH) / Path("Signals/Algo1/male_inst_sim_3.wav"),
                Path(SASSEC_PATH) / Path("Signals/Algo1/male_inst_sim_4.wav"),
            ],
        },
    },
]