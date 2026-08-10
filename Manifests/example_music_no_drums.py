import os
from pathlib import Path

# Set SASSEC_PATH to the extracted dataset root, or place SASSEC in the current directory.
SASSEC_PATH = Path(os.environ.get("SASSEC_PATH", "SASSEC")).expanduser()

MANIFEST = [
    {
        "mixture_id": "female_pair_example",
        "references": [
            SASSEC_PATH / "Signals/orig/nodrums_inst_sim_1.wav",
            SASSEC_PATH / "Signals/orig/nodrums_inst_sim_2.wav",
            SASSEC_PATH / "Signals/orig/nodrums_inst_sim_3.wav",
        ],
        "systems": {
            "Algo1_SASSEC": [
                SASSEC_PATH / "Signals/Algo1/nodrums_inst_sim_1.wav",
                SASSEC_PATH / "Signals/Algo1/nodrums_inst_sim_2.wav",
                SASSEC_PATH / "Signals/Algo1/nodrums_inst_sim_3.wav",
            ],
        },
    },
]
