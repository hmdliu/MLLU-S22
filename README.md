# MLLU-S22 Final Project Codebase

**Codebase adopted from [OpenDelta](https://github.com/thunlp/OpenDelta).**

---

## Group Info
Members:
- Haoming(Hammond) Liu \
- Xiaochen(Nigel) Lu \
- Wenbin(Jim) Qi \

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
git clone https://github.com/hmdliu/MLLU-S22
cd MLLU-S22
```

---

## Test Run
```
cd /scratch/$USER/MLLU-S22
sbatch run_seq2seq.slurm configs/none/mnli.json
```
