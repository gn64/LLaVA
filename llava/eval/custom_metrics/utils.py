import torch
from PIL import Image
from transformers import (
    LlavaForConditionalGeneration,
    LlavaNextForConditionalGeneration,
)
from peft import PeftModel
from numpy import asarray
import numpy as np
from tqdm import tqdm
import os
import json
import shutil
import safetensors.torch


def append_to_json(file_path, epoch_data, epoch):
    if os.path.exists(file_path):
        with open(file_path, "r+") as f:
            data = json.load(f)
            data[str(epoch)] = epoch_data
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=4)
    else:
        with open(file_path, "w") as f:
            json.dump({str(epoch): [epoch_data]}, f, indent=4)


def process_inference_for_single_instruction(
    model,
    processor,
    data_loader,
    user_prompt="USER: \n",
    assistant_prompt="ASSISTANT:",
    max_new_token=50,
    process_batch_num=None,
    **kwargs,
):
    """
    Process inference for a single instruction.

    Args:
        model (object): The model used for inference.
        processor (object): The processor used for data preprocessing.
        data_loader (object): The data loader containing the input data.
        user_prompt (str, optional): The user prompt for the instruction. Defaults to "USER: <image>\n".
        assistant_prompt (str, optional): The assistant prompt for the instruction. Defaults to "ASSISTANT:".
        max_new_token (int, optional): The maximum number of new tokens to generate. Defaults to 200.
        process_batch_num (int, optional): Limit the number of batches to process. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the processor.

    Returns:
        list: A list of dictionaries containing the generated output, instruction, answer, and other optional fields.
    """
    ret = []
    model.eval()
    for batch_i, batch in enumerate(tqdm(data_loader)):
        # print(f"{batch.keys()}")
        if process_batch_num:
            if batch_i >= process_batch_num:
                break
        prompt = [
            f"{user_prompt}{instr[0]['value']}\n{assistant_prompt}"
            for instr in batch["conversations"]
        ]
        img = [
            asarray(Image.open(img_path).convert("RGB")).transpose(2, 0, 1)
            for img_path in batch["image"]
        ]
        inputs = processor(
            prompt, images=img, padding=True, return_tensors="pt", **kwargs
        ).to("cuda", torch.bfloat16)
        model.to("cuda")
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + max_new_token,
            **kwargs,
            do_sample=False,
        )
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)
        for i, text in enumerate(generated_text):
            ans = {}
            ans["output"] = text.split("ASSISTANT:")[-1]
            ans["instr"] = batch["conversations"][i][0]["value"]
            ans["answer"] = batch["conversations"][i][1]["value"]
            if "id" in batch:
                ans["id"] = batch["id"][i]
            if "idx" in batch:
                ans["idx"] = batch["idx"][i]
            if "img_path" in batch:
                ans["img_path"] = batch["img_path"][i]
            if "labels" in batch:
                ans["labels"] = batch["labels"][i]
            if "boxes" in batch:
                ans["boxes"] = batch["boxes"][i]
            if "pathologies" in batch:
                ans["pathologies"] = batch["pathologies"][i]
            ret.append(ans)
            # print(ans.keys())
    return ret


def remap_lora_keys(lora_weights):
    new_weights = {}
    # Iterate over the keys and modify them if necessary
    for key, value in lora_weights.items():
        if key.startswith("base_model.model.model.vision_tower.vision_tower."):
            new_key = key.replace(
                "base_model.model.model.vision_tower.vision_tower.",
                "base_model.model.vision_tower.",
            )
            new_weights[new_key] = value
        elif key.startswith("model.image_newline"):
            continue
        elif key.startswith("base_model.model.lm_head.weight"):
            new_key = "base_model.model.language_model.lm_head.weight"
            new_weights[new_key] = value
        elif key.startswith("base_model.model.model."):
            new_key = key.replace(
                "base_model.model.model.", "base_model.model.language_model.model."
            )
            new_weights[new_key] = value
        else:
            new_weights[key] = value
    return new_weights


def remap_keys_nonlora(old_key):
    # Strip the leading 'base_model.' if present
    if old_key.startswith("base_model."):
        old_key = old_key[11:]

    # Replace the nested 'model.model.mm_projector' with 'multi_modal_projector'
    # and correct the linear layer names
    old_key = old_key.replace(
        "model.model.mm_projector.", "multi_modal_projector.linear_"
    )

    # Correct the numbering (0 becomes 1, 2 becomes 2)
    old_key = old_key.replace("linear_0", "linear_1")
    old_key = old_key.replace("linear_2", "linear_2")

    return old_key


def load_llava_with_lora(model_id_or_path, checkpoint_lora_path):

    config_path_lora = os.path.join(checkpoint_lora_path, "adapter_config.json")
    config_path_lora_backup = os.path.join(
        checkpoint_lora_path, "adapter_config_backup.json"
    )
    shutil.copyfile(config_path_lora, config_path_lora_backup)
    with open(config_path_lora, "r") as file:
        config = json.load(file)
    config["base_model_name_or_path"] = model_id_or_path
    with open(config_path_lora, "w") as file:
        json.dump(config, file, indent=4)
    print("Configuration updated successfully!")

    # Load the base model
    if "1.5" in model_id_or_path:
        model = LlavaForConditionalGeneration.from_pretrained(model_id_or_path)
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(model_id_or_path)
    non_lora_path = os.path.join(checkpoint_lora_path, "non_lora_trainables.bin")
    non_lora_weights = torch.load(non_lora_path)
    print("Non-LoRA weights loaded.")
    non_lora_weights = {remap_keys_nonlora(k): v for k, v in non_lora_weights.items()}
    model.load_state_dict(non_lora_weights, strict=False)
    print("Non-LoRA weights applied to the model.")

    lora_path = os.path.join(checkpoint_lora_path, "adapter_model.safetensors")
    lora_backup_path = os.path.join(
        checkpoint_lora_path, "adapter_model_backup.safetensors"
    )
    lora_weights = safetensors.torch.load_file(lora_path)
    remapped_lora_weights = remap_lora_keys(lora_weights)
    model.load_state_dict(remapped_lora_weights, strict=False)
    shutil.copyfile(lora_path, lora_backup_path)
    safetensors.torch.save_file(remapped_lora_weights, lora_path)
    # print(f"Remapped weights have been saved to {lora_path}.")
    print("Loading LoRA weights...")
    model = PeftModel.from_pretrained(model, checkpoint_lora_path)
    print("Merging LoRA weights...")
    model = model.merge_and_unload()
    print("Model is loaded...")
    shutil.copyfile(config_path_lora_backup, config_path_lora)
    print("adapter_config.json restored.")
    shutil.copyfile(lora_backup_path, lora_path)
    print("adapter_model.bin restored.")

    return model


def custom_collate_fn(batch):
    # Initialize a dictionary to hold the collated data
    collated_batch = {}

    # Assuming all items in the batch have the same structure
    # Loop over the keys in the first item of the batch to get all attributes
    for key in batch[0]:
        # Collect data for each attribute across all items in the batch
        collated_batch[key] = [item[key] for item in batch]

    return collated_batch
