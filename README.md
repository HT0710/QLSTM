# <p align="center">Modified Quantum Long-Short Term Memory with Variational Quantum Circuits for PV Power Forecasting</p>
![CLSTM drawio](https://github.com/user-attachments/assets/8ef76e0c-c6de-4a5b-9c9d-4111b9c0b2e9)

## Table of Contents
- [Modified Quantum Long-Short Term Memory with Variational Quantum Circuits for PV Power Forecasting](#modified-quantum-long-short-term-memory-with-variational-quantum-circuits-for-pv-power-forecasting)
  - [Introduction](#introduction)
  - [Requirements](#requirements)
  - [Configuration](#configuration)
  - [Train](#train)
  - [Logging](#logging)
  - [Demo App](#demo-app)
  - [License](#license)
  - [Contact](#contact)

## Introduction
This repository presents a hybrid AI model that integrates Quantum Machine Learning (QML) and Deep Learning to deliver fast, cost-efficient, and highly accurate photovoltaic (PV) power forecasting. By embedding Variational Quantum Circuits (VQCs) within a modified Long Short-Term Memory (LSTM) framework, we propose a model that not only improves prediction accuracy but also significantly reduces computational resources and time compared to classical ML and deep learning counterparts.

## Requirements
Project is tested with **Python >= 3.10**

```bash
pip install -r requirements.txt
```

## Configuration
Most of the configuration for this project can be changed in `/qlstm/configs/`:  
- `train.yaml` for training
- `callbacks.yaml` for logging and training behavior

**To change training model**:
1. Use or create new model class in `qlstm/models`
2. Import and define your model at line 42 from `qlstm/train.py`.
3. Change model config in `qlstm/configs/train.yaml`

**For datasets with difference structure**:
1. Create a custom data module at `qlstm/modules/data.py` inherited from `CustomDataModule` class.  
2. Import and define your custom data module at line 39 from `qlstm/train.py`.
3. Change data config in `qlstm/configs/train.yaml`

## Train
Modify the configurations in `qlstm/configs/train.yaml` then run:
```bash
python qlstm/train.py
```
  
With CLI:
```bash
python qlstm/train.py -h
```
Example:
```bash
py qlstm/train.py trainer.accelerator=cpu optimizer=0.001
```

## Logging
Note: Use full path
```bash
tensorboard --logdir /home/USER/.../QLSTM/lightning_logs
```

## Demo App
Launch local:
```bash
python app/app.py
```

Launch public:
```bash
python app/app.py --share
# or
python app/app.py -s
```

## License
This project is licensed under the MIT License. See [LICENSE](https://github.com/HT0710/QLSTM/blob/main/LICENSE) for more details.

## Contact
Open an issue: [New issue](https://github.com/HT0710/QLSTM/issues/new)

Mail: pthung7102002@gmail.com

---
