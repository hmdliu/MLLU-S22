# MLLU Final Project Codebase

**Adopted from [Compacter](https://github.com/rabeehk/compacter) and [OpenDelta](https://github.com/thunlp/OpenDelta).**

---

## Group Members
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
We test the [OpenDelta](https://github.com/thunlp/OpenDelta) implementation of delta tuning methods on the [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/) dataset.
```
# switch to project root dir
cd /scratch/$USER/MLLU-S22

# seq2seq training based on ./configs/[method]/[dataset].json
sbatch run_seq2seq.slurm [method] [dataset]

# check val & test results (after the job ends)
cat log/[method]/[dataset]/results.jsonl
```
### Available methods
- **Fine-tuning**: none
- [**Adapter**](https://arxiv.org/abs/1902.00751): adapter
- [**BitFit**](https://arxiv.org/abs/2106.10199): bitfit
- [**LoRA**](https://arxiv.org/abs/2106.09685): lora
- [**Prefix-tuning**](https://arxiv.org/abs/2101.00190): prefix
- [**Soft-prompt Tuning**](https://arxiv.org/abs/2104.08691): ~~soft_prompt~~ (buggy)

### Available datasets
- [**MultiNLI**](https://cims.nyu.edu/~sbowman/multinli/): mnli
- *More datasets will be added soon.*