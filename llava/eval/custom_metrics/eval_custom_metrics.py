import argparse
import torch
from transformers import AutoProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
from utils import (
    process_inference_for_single_instruction,
    custom_collate_fn,
    append_to_json,
    load_llava_with_lora,
)
from compute_metrics_tasks import process_match_classification_with_metrics
from transformers import BitsAndBytesConfig
from PIL import Image
import os
import re


def get_checkpoint_path(checkpoint_folder, evaluate_best_epoch):
    if evaluate_best_epoch:
        checkpoint_path = os.path.join(checkpoint_folder, "best_llava_eval_model")
    else:
        subfolders = [f.name for f in os.scandir(checkpoint_folder) if f.is_dir()]
        checkpoint_regex = re.compile(r"^checkpoint-(\d+)$")
        max_checkpoint = None
        max_num = -1

        for folder in subfolders:
            match = checkpoint_regex.match(folder)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
                    max_checkpoint = folder

        if max_checkpoint is not None:
            checkpoint_path = os.path.join(checkpoint_folder, max_checkpoint)
        else:
            ValueError("No checkpoint folder found in the directory.")
    return checkpoint_path


def main(args):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
    )

    processor = AutoProcessor.from_pretrained(args.model_id)
    checkpoint = get_checkpoint_path(args.checkpoint_folder, args.evaluate_best_epoch)
    print(f"Load checkpoint: {checkpoint}")
    model = load_llava_with_lora(args.model_id, checkpoint)
    test_dataset = load_dataset("json", data_files={"test": args.data_path})["test"]
    data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn,
    )

    with torch.no_grad():
        dataframe = process_inference_for_single_instruction(
            model, processor, data_loader
        )
        dataframe_modified = process_match_classification_with_metrics(
            dataframe, labels=None
        )
        print(dataframe_modified["metrics"])
        dataframe_modified["epoch"] = args.epoch
        append_to_json(args.metrics_output_path, dataframe_modified, args.epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference and process metrics with a LLAMA model."
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="Model identifier for HF repository.",
    )
    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        required=True,
        help="Path to the checkpoint directory.",
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the dataset JSON file."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for data loading."
    )
    parser.add_argument(
        "--metrics_output_path",
        type=str,
        required=True,
        help="Path to output JSON file for metrics.",
    )
    parser.add_argument("--epoch", type=int, default=1, help="Current epoch number.")
    parser.add_argument(
        "--evaluate_best_epoch",
        type=bool,
        default=False,
        help="Evaluate the best epoch.",
    )

    args = parser.parse_args()
    main(args)
