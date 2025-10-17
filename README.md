# 🪃 Boomerang Distillation

Boomerang distillation is a phenomenon in LLMs, where distilling a teacher model into a student model enables us to reconstruct intermediate-sized models by incorporating teacher layers into the student with no additional training.

This repo contains code for boomerang distillation from our paper [Boomerang Distillation Enables Zero-Shot Model Size Interpolation](https://arxiv.org/abs/2510.05064).

## Table of contents
- [Boomerang Distillation](#boomerang-distillation)
    - [Table of contents](#table-of-contents)
    - [Installation](#installation)
    - [Distillation Training](#distillation-training-optional)
    - [Patching and Evaluation](#patching-and-evaluation)
    - [Notebooks](#notebooks)
    - [Citation](#citation)


## Installation

To install all of the required packages, run the following:
```
conda create -n boomerang-distillation python==3.12
conda activate boomerang-distillation
pip3 install -r requirements.txt
```
To reproduce the environment used in the paper experiments, use `requirements_dev.txt`. We note that the requirements have only been tested in linux-based systems and may be unsupported for other operating systems.

## Distillation training (optional)

We provide the distilled student models used in our paper on [Hugging Face](https://huggingface.co/collections/Harvard-DCML/boomerang-distillation-68e95c276a09358d9a39b52e). These models can directly be loaded and patched with their corresponding teacher blocks to create intermediate models.

If you wish to train custom student models using `train/train.py`, the following training script will train a student model pruned and distilled from `Qwen3-4B-Base` using 4 GPUs (make sure you installed `requirements_dev.txt`):

```bash
TEACHER="Qwen/Qwen3-4B-Base"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc-per-node=4 --module train.train \
    --teacher_model_name_or_path $TEACHER \
    --save_directory "/path/to/save/directory" \ # add your save directory here
    --dataset "EleutherAI/the_pile_deduplicated" \
    --fsdp_config $TEACHER \
```
Options:
- `teacher_model_name_or_path`: Hugging Face reference or local model path for the teacher model.
- `save_directory`: directory to save distilled models in.
- `dataset`: dataset used for distillation. We use the deduplicated version of The Pile in our paper. Note that by default, distillation is run for `500` steps consisting of 4.2M tokens to match the setting from our paper, but this can be changed by setting `--max_steps`.
- `fsdp_config`: if using fsdp, set this to the teacher model name to ensure that fsdp chooses the correct modules to wrap (see `train/training_args.py`)

Notes:
1. The training and initialization hyperparameters and dataset are set by default to those used in our paper, but they can be edited to your specifications. The arguments are documented in `train/training_args.py`, `train/model_args.py`, and `train/data_args.py`.
2. The training script supports models from the `Qwen`, `Pythia`, and `Llama` families. The initialization code may need to be adjusted for other models.

## Patching and Evaluation

Given a distilled student and teacher model pair, you can construct intermediate-sized models using the `build_intermediate_model` function from `patching/patch.py`. To create intermediate models and evaluate them with `lm-evaluation-harness`, run the script in `evaluate/evaluate.py` as follows:

```bash
TEACHER="Qwen/Qwen3-4B-Base"
STUDENT="Harvard-DCML/boomerang-qwen3-2.3B"

python3 -m evaluate.evaluate \
    --teacher_model_name_or_path $TEACHER \
    --student_model_name_or_path $STUDENT \
    --save_directory "/path/to/save/directory" \
    --num_layers_to_patch 4 \
```
Options:
- `teacher_model_name_or_path`: Hugging Face reference or local model path for the teacher model.
- `student_model_name_or_path`: Hugging Face reference or local model path for the student model. The model paths we provide on Hugging Face are in the table below.
- `save_directory`: local folder to save `lm-evaluation-harness` results in.
- `num_layers_to_patch`: number of student layers to patch with their corresponding teacher blocks. The minimum and maximum values for each model are in the table below.
- `patch_first_k_layers`: include this argument to patch the first `num_layers_to_patch` student layers, otherwise the last `num_layers_to_patch` layers will be patched. This defaults to True for `Llama-3.2-3B` and False for the remaining models.
- `tasks`: comma-separated string of `lm-evaluation-harness` tasks to evaluate the intermediate model on. Set to the full suite of tasks used in the paper (`"arc_easy,arc_challenge,boolq,hellaswag,openbookqa,piqa,winogrande,race,mmlu,rte,wikitext,gsm8k_cot,ifeval,hendrycks_math"`) by default.
- `eval_batch_size`: batch size used for evaluation (default `4`).
-  `dtype`: data type used to load the model weights. Set to `bfloat16` by default.
- `override_llama_patching`: if set, overrides the default patching order for Llama models (first k layers) and uses the order specified by `patch_first_k_layers`.

#### Model configurations

| `teacher_model_name_or_path` | `student_model_name_or_path` | Range of `num_layers_to_patch` |
|:---:|:---:|:---:|
| Qwen/Qwen3-4B-Base | Harvard-DCML/boomerang-qwen3-2.3B | 1-17 |
| Qwen/Qwen3-8B-Base | Harvard-DCML/boomerang-qwen3-4.9B  | 1-17 |
| EleutherAI/pythia-2.8b | Harvard-DCML/boomerang-pythia-1.6B | 1-15 |
| EleutherAI/pythia-6.9b | Harvard-DCML/boomerang-pythia-3.8B | 1-15 |
| meta-llama/Llama-3.2-3B | Harvard-DCML/boomerang-llama-3.2-1.9B | 1-13 |


## Notebooks

We provide an example notebook for reproducing our DistilBERT results in `notebooks/test_distilbert.ipynb` and on Google Colab at [test_distilbert.ipynb](https://drive.google.com/file/d/1bAzX436ZH4zQmk5iQNauAOhGHIBJ1CkB/view?usp=sharing). The DistilBERT experiments can be replicated on the T4 GPUs provided in Colab.

## Citation

If you use boomerang distillation in your work, please cite our paper:
```
@article{kangaslahti2025boomerang,
  title={Boomerang Distillation Enables Zero-Shot Model Size Interpolation},
  author={Kangaslahti, Sara and Nayak, Nihal V and Geuter, Jonathan and Fumero, Marco and Locatello, Francesco and Alvarez-Melis, David},
  journal={arXiv preprint arXiv:2510.05064},
  year={2025},
  url={https://arxiv.org/abs/2510.05064}
}
```
