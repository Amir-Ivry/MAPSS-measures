from pathlib import Path

SASSEC_PATH = "C:/postdoc/R1 - PS and PM/Data - speech separation MOS/SASSEC"  # full path

MANIFEST = [
    {
        "mixture_id": "female_pair_example",
        "references": [
            Path(SASSEC_PATH) / Path("Signals/orig/nodrums_inst_sim_1.wav"),
            Path(SASSEC_PATH) / Path("Signals/orig/nodrums_inst_sim_2.wav"),
            Path(SASSEC_PATH) / Path("Signals/orig/nodrums_inst_sim_3.wav"),
        ],
        "systems": {
            "Algo1_SASSEC": [
                Path(SASSEC_PATH) / Path("Signals/Algo1/nodrums_inst_sim_1.wav"),
                Path(SASSEC_PATH) / Path("Signals/Algo1/nodrums_inst_sim_2.wav"),
                Path(SASSEC_PATH) / Path("Signals/Algo1/nodrums_inst_sim_3.wav"),
            ],
        },
    },
]