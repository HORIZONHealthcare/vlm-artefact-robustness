# Medical Image Artefact Robustness in Vision-Language Models

Code for *Towards Generalisable Robustness to Medical Image Artefacts in Vision-Language Models: Efficient Prompting and Adaptation Strategies*.

![Study overview](images/overview.png)

## Code and data

- [Representative study images](data%20samples/README.md)
- [Actual few-shot cases](fewshot_case/README.md): four conditions with fixed image ordering
- [Preprocessing](preprocessing/): resizing and artefact generation
- [Training](training/): projector, LoRA, and final-representation adapter tuning
- [Evaluation](evaluation/): open-weight baseline, four prompt configurations, few-shot, adapter, projector, and LoRA inference

Open-weight backends: `qwen3`, `qwen35`, `internvl`, `medgemma`, `lingshu`, and `lingshu_i`. API scripts cover GPT-5.5 and Gemini 3.1 Pro.

## Quick start

Run from the repository root using Python 3.11 and a suitable GPU environment:

```bash
python -m pip install -r requirements.txt
python -m evaluation.vlm_evaluation \
  --run_vlm_baseline --backend qwen3 \
  --data_csv metadata/test.csv --data_root data/ \
  --out_json outputs/qwen3_internal_zero_shot.json --seed 42
```

Prepare your own CSV using [example_train_metadata.csv](example_train_metadata.csv): `dataset` (`mri`, `oct`, `xray`), `filepath`, `fileindex`, `use_type` (`train`, `val`, `test`), `severity` (`clean`, `weak`), `binarylabel` (0 normal, 1 disease), and `corruption`. Paths are absolute or relative to `--data_root`; all views of a source image share its ID and split. Evaluation uses test rows.

Optional dependencies: [4-bit loading](requirements-4bit.txt) and [commercial APIs](requirements-commercial.txt). API scripts use `OPENAI_API_KEY` or `GEMINI_API_KEY` from the environment.

Select `--prompt_config base`, `disease`, `artefact`, or `combined` for zero-shot prompting.

## Training and few-shot inference

Train an adapter on clean and corrupted training/validation rows:

```bash
python -m training.adapter_train \
  --backend qwen3 --clean_csv metadata/clean.csv \
  --artefact_csv metadata/artefact.csv --data_root data/ \
  --save_dir outputs/adapter --tag seed42 --seed 42
```

Evaluate it with `evaluation.vlm_evaluation --run_vlm --adapter_on_clean --vlm_ckpt outputs/adapter/qwen3_seed42_best.pt`, adding the same backend, test CSV, data root, and output arguments as above.

```bash
python -m evaluation.vlm_few_shot --backend qwen3 \
  --demo_csv fewshot_case/few_shot_2.csv --data_root . \
  --data_csv metadata/test.csv --out_json outputs/qwen3_fewshot.json
```

The few-shot command uses repository-relative paths; query paths must be absolute or start with `data/` when stored there. Use `--query_split val` for configuration selection and `test` for held-out evaluation.

Projector and LoRA training use `training.proj_gen_train` and `training.lora_gen_train`; checkpoint inference uses `evaluation.generative_adaptation`. Preprocessing uses `preprocessing.apply_resize`, `apply_artefacts`, and `apply_new_artefacts`. Each module lists arguments through `python -m <module> --help`. Pin the same `--model_revision` in open-weight training and evaluation.

## Data availability

The datasets used in this study are derived from publicly available sources:

| Modality | Internal dataset | External dataset |
| --- | --- | --- |
| Brain MRI | [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | [Mendeley Data, version 1](https://data.mendeley.com/datasets/b3z27t8vpb/1) |
| Retinal OCT | [Mendeley Data, version 3](https://data.mendeley.com/datasets/rscbjbr9sj/3) | [Mendeley Data, version 2](https://data.mendeley.com/datasets/8kt969dhx6/2); [OCT and Eye Fundus Dataset repository](https://github.com/Traslational-Visual-Health-Laboratory/OCT-AND-EYE-FUNDUS-DATASET) |
| Chest X-ray | [COVID-19 Detection X-Ray Dataset (Kaggle)](https://www.kaggle.com/datasets/darshan1504/covid19-detection-xray-dataset) | [NIH Chest X-rays (Kaggle)](https://www.kaggle.com/datasets/nih-chest-xrays/data) |

COVID-19 images were excluded from the internal chest X-ray cohort. Representative images and actual few-shot examples are included; obtain full datasets separately from their providers. 

Questions: [GitHub Issues](https://github.com/HORIZONHealthcare/vlm-artefact-robustness/issues).
