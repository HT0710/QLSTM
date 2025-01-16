# QLSTM


### Requirements
##### Python >= 3.10
```bash
pip install -r requirements.txt
```

### Configuration
Change configuration in `/qlstm/configs/...`
For training: `train.yaml`
For logging: `callbacks.yaml`

### Train
```bash
python qlstm/train.py
```

### Logging
Tensorboard:
Note: Use full path
```bash
tensorboard --logdir /home/USER/.../QLSTM/lightning_logs
```
