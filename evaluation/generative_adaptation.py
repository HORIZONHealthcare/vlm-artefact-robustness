"""Held-out inference with the released projector and LoRA checkpoints."""
import argparse
import importlib
from pathlib import Path

import torch
from repro_utils import build_repro_metadata, resolved_model_revision, save_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=["projector", "lora"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--data_root", default=".")
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--model_revision")
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    cli = parser.parse_args()
    if Path(cli.out_json).exists() and not cli.overwrite:
        raise FileExistsError("Output exists; choose a new path or pass --overwrite")
    trainer = importlib.import_module("training.proj_gen_train" if cli.method == "projector" else "training.lora_gen_train")
    checkpoint = trainer.load_trusted_checkpoint(cli.checkpoint)
    backend = checkpoint["backend"]
    args = trainer.parse_args(["--backend", backend, "--clean_csv", cli.data_csv, "--artefact_csv", cli.data_csv])
    for key, value in checkpoint.get("config", {}).items():
        if hasattr(args, key):
            setattr(args, key, value)
    args.backend = backend
    revision = checkpoint.get("reproducibility", {}).get("model", {}).get("resolved_revision") or args.model_revision
    if cli.model_revision and revision and cli.model_revision != revision:
        raise ValueError("Requested model revision differs from the checkpoint")
    args.model_revision = cli.model_revision or revision
    trainer.set_seed(cli.seed, deterministic=not args.allow_nondeterministic)
    trainer.configure_hf_cache_portable(args.hf_cache_dir)
    if cli.method == "projector":
        trainer.configure_hf_loading(args)
    model_id = checkpoint.get("model_id") or trainer.MODEL_ID_BY_BACKEND[backend]
    frame = trainer.load_split_csv(cli.data_csv, cli.data_root, default_severity="clean", check_paths=True, allowed_splits=("test",))
    frame = frame[frame.use_type_norm.eq("test")].copy()
    if frame.empty:
        raise ValueError("No held-out test rows")
    model, processor = trainer.load_backend(backend, model_id, args)
    if cli.method == "projector":
        projector, _ = trainer.freeze_all_but_projector(model, backend, checkpoint.get("projector_path"))
        projector.load_state_dict(checkpoint["projector"], strict=True)
    else:
        targets = checkpoint.get("lora_targets") or trainer.parse_lora_targets(args.lora_targets)
        names = trainer.inject_lora_adapters(model, target_names=targets, r=args.lora_r,
                                             alpha=args.lora_alpha, dropout=args.lora_dropout)
        if checkpoint.get("lora_module_names") and names != checkpoint["lora_module_names"]:
            raise ValueError("LoRA module structure differs from the checkpoint")
        if set(trainer.collect_lora_state(model)) != set(checkpoint["lora"]):
            raise ValueError("LoRA checkpoint parameter keys differ")
        model.load_state_dict(checkpoint["lora"], strict=False)
    model.eval()
    dataset = trainer.CleanMultiWeakDataset(frame, trainer.PROMPT_BY_DATASET, trainer.SYSTEM_PROMPT_SHORT,
                                           max_weaks=5, normal_text="normal", disease_text="disease")
    collate = trainer.make_generative_collate_fn(processor, backend, include_prompt_inputs=True)
    records = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            batch = collate([dataset[index]])
            texts, predictions = trainer.generate_batch_predictions(model, processor, batch, cli.max_new_tokens)
            for i, (text, prediction) in enumerate(zip(texts, predictions)):
                records.append(dict(dataset=batch["modality"][i], fileindex=batch["fileindices"][i],
                                    filepath=batch["paths"][i], y_true=int(batch["labels_cls"][i]),
                                    pred=int(prediction), raw_text=text,
                                    severity="clean" if bool(batch["is_clean"][i]) else "weak",
                                    corruption=batch["corruption"][i], use_type="test"))
    if len(records) != len(frame):
        raise ValueError("Incomplete source-image groups or more than five artefact views; no partial output saved")
    repro = build_repro_metadata(script_path=__file__, args=cli, input_files=[cli.data_csv, cli.checkpoint],
                                model_id=model_id, requested_revision=args.model_revision,
                                resolved_revision=resolved_model_revision(model, args.model_revision),
                                prompts={"system": trainer.SYSTEM_PROMPT_SHORT, "by_dataset": trainer.PROMPT_BY_DATASET})
    Path(cli.out_json).parent.mkdir(parents=True, exist_ok=True)
    save_json_atomic(cli.out_json, {"method": cli.method, "backend": backend,
                                   "per_image": records, "reproducibility": repro})


if __name__ == "__main__":
    main()
