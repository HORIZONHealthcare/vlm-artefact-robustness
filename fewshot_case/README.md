# Actual few-shot demonstration sets

The four CSVs use the actual supplied study examples. Paths are relative to the repository root; run inference with `--data_root .`. Row ordering is preserved from the supplied source CSVs.

| CSV | Cases per modality | Images per modality | Total images |
| --- | --- | --- | --- |
| few_shot_1_n.csv | One normal | 6 | 18 |
| few_shot_1_d.csv | One disease | 6 | 18 |
| few_shot_2.csv | One normal and one disease | 12 | 36 |
| few_shot_4.csv | Two normal and two disease | 24 | 72 |

Each case has one clean and five seen-artefact views and is labelled `use_type=train`. Images are organised under condition/modality/label/clean or artefacts. In the original four-case CSV, five X-ray rows had shifted artefact names and filenames. The portable CSV corrects these metadata fields to match each actual `filepath`, preserving the original image sequence and image bytes.

```bash
python -m evaluation.vlm_few_shot --backend qwen3 \
  --demo_csv fewshot_case/few_shot_2.csv --data_root . \
  --data_csv metadata/test.csv --out_json outputs/qwen3_fewshot.json
```

Keep query paths absolute or relative to the chosen data root. The scripts validate training-only complete cases and the four study class compositions.
