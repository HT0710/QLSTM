import rootutils
from lightning.pytorch import seed_everything

rootutils.autosetup(".gitignore")

from qlstm.models.cLSTM import cLSTM
from qlstm.models.cQLSTM import cQLSTM
from qlstm.models.LSTM import LSTM
from qlstm.models.QLSTM import QLSTM

seed_everything(42)

MODELS = {
    "QLSTM": {
        "init": QLSTM(9, 128, n_qubits=4),
        "checkpoint": "lightning_logs/QLSTM/4q/checkpoints/last.ckpt",
    },
    "LSTM": {
        "init": LSTM(9, 128),
        "checkpoint": "lightning_logs/LSTM/base/checkpoints/last.ckpt",
    },
    "cLSTM": {
        "init": cLSTM(9, 128),
        "checkpoint": "lightning_logs/cLSTM/base/checkpoints/last.ckpt",
    },
    "cQLSTM": {
        "init": cQLSTM(9, 128, n_qubits=4),
        "checkpoint": "lightning_logs/cQLSTM/4q_post_2/checkpoints/last.ckpt",
    },
}
