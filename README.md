# MLLU-S22 Final Project Codebase

**Codebase adopted from [Compacter](https://github.com/rabeehk/compacter) and [OpenDelta](https://github.com/thunlp/OpenDelta).**

---

## Group Info
Members:
- Haoming(Hammond) Liu
- Xiaochen(Nigel) Lu
- Wenbin(Jim) Qi

---

## Requisites
- Test Env: Python 3.9.7 (Singularity)
- Major Packages:
    - torch (1.10.2+cu113), transformers (4.18.0), datasets(2.1.0)
    - **opendelta (0.0.4)**

---

## Clone codebase
```
cd /scratch/$USER
git clone https://github.com/hmdliu/MLLU-S22 && cd MLLU-S22
```

---

## Test Run
We test the [OpenDelta](https://github.com/thunlp/OpenDelta) implementation of delta tuning methods on the [MNLI](https://cims.nyu.edu/~sbowman/multinli/) dataset.
```
cd /scratch/$USER/MLLU-S22
sbatch run_seq2seq.slurm [config_path]
```
Available Configs:
- **Baseline**: configs/none/mnli.json
- **Adapter**: configs/adapter/mnli.json
- **BitFit**: configs/bitfit/mnli.json
- **LoRA**: configs/lora/mnli.json
- **Prefix-Tuning**: configs/prefix/mnli.json
- **Soft-prompt Tuning**: ~~configs/soft_prompt/mnli.json~~ (buggy)

**More datasets will be added soon.**