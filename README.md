# MLLU Final Project Codebase

**DS-UA 203: Machine Learning for Language Understanding (Spring 2022)**

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

## Run Experiments
```
# switch to project root dir
cd /scratch/$USER/MLLU-S22

# seq2seq training args:
#  - dataset: a dataset to be trained and evaluated on
#  - delta_type: a delta tuning method to be applied
#  - data_ratio: ratio of training samples (between 0 and 1)
sbatch run_seq2seq.slurm [dataset] [delta_type] [data_ratio]

# check val & test results (after the job ends)
cat log/[dataset]/[delta_type]/results.jsonl
```
### Available datasets
- [**Yelp Polarity**](https://huggingface.co/datasets/yelp_polarity): yelp
- [**MultiNLI**](https://huggingface.co/datasets/multi_nli): mnli
- [**SQuAD**](https://huggingface.co/datasets/squad): squad
- [**RACE**](https://huggingface.co/datasets/race): race

### Available delta types
- **Fine-tuning**: none
- [**Adapter**](https://arxiv.org/abs/1902.00751): adapter
- [**BitFit**](https://arxiv.org/abs/2106.10199): bitfit
- [**LoRA**](https://arxiv.org/abs/2106.09685): lora
- [**Prefix-tuning**](https://arxiv.org/abs/2101.00190): prefix
