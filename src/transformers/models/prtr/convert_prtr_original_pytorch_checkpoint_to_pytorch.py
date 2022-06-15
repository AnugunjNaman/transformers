# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert PRTR checkpoints."""


import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image

import requests
from huggingface_hub import hf_hub_download
from transformers import PrtrConfig, PrtrFeatureExtractor, PrtrForObjectDetection
from transformers.utils import logging

from models.pose_transformer import get_pose_net
from config import cfg
cfg.merge_from_file('config/res50_384x288_adamw_lr1e-4.yaml')


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# here we list all keys to be renamed (original name on the left, our name on the right)
rename_keys = []
for i in range(6):
    # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.self_attn.out_proj.weight", f"encoder.layers.{i}.self_attn.out_proj.weight")
    )
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.self_attn.out_proj.bias", f"encoder.layers.{i}.self_attn.out_proj.bias")
    )
    rename_keys.append((f"transformer.encoder.layers.{i}.linear1.weight", f"encoder.layers.{i}.fc1.weight"))
    rename_keys.append((f"transformer.encoder.layers.{i}.linear1.bias", f"encoder.layers.{i}.fc1.bias"))
    rename_keys.append((f"transformer.encoder.layers.{i}.linear2.weight", f"encoder.layers.{i}.fc2.weight"))
    rename_keys.append((f"transformer.encoder.layers.{i}.linear2.bias", f"encoder.layers.{i}.fc2.bias"))
    rename_keys.append(
        (f"transformer.encoder.layers.{i}.norm1.weight", f"encoder.layers.{i}.self_attn_layer_norm.weight")
    )
    rename_keys.append((f"transformer.encoder.layers.{i}.norm1.bias", f"encoder.layers.{i}.self_attn_layer_norm.bias"))
    rename_keys.append((f"transformer.encoder.layers.{i}.norm2.weight", f"encoder.layers.{i}.final_layer_norm.weight"))
    rename_keys.append((f"transformer.encoder.layers.{i}.norm2.bias", f"encoder.layers.{i}.final_layer_norm.bias"))
    # decoder layers: 2 times output projection, 2 feedforward neural networks and 3 layernorms
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.self_attn.out_proj.weight", f"decoder.layers.{i}.self_attn.out_proj.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.self_attn.out_proj.bias", f"decoder.layers.{i}.self_attn.out_proj.bias")
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.multihead_attn.out_proj.weight",
            f"decoder.layers.{i}.encoder_attn.out_proj.weight",
        )
    )
    rename_keys.append(
        (
            f"transformer.decoder.layers.{i}.multihead_attn.out_proj.bias",
            f"decoder.layers.{i}.encoder_attn.out_proj.bias",
        )
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.linear1.weight", f"decoder.layers.{i}.fc1.weight"))
    rename_keys.append((f"transformer.decoder.layers.{i}.linear1.bias", f"decoder.layers.{i}.fc1.bias"))
    rename_keys.append((f"transformer.decoder.layers.{i}.linear2.weight", f"decoder.layers.{i}.fc2.weight"))
    rename_keys.append((f"transformer.decoder.layers.{i}.linear2.bias", f"decoder.layers.{i}.fc2.bias"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm1.weight", f"decoder.layers.{i}.self_attn_layer_norm.weight")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.norm1.bias", f"decoder.layers.{i}.self_attn_layer_norm.bias"))
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm2.weight", f"decoder.layers.{i}.encoder_attn_layer_norm.weight")
    )
    rename_keys.append(
        (f"transformer.decoder.layers.{i}.norm2.bias", f"decoder.layers.{i}.encoder_attn_layer_norm.bias")
    )
    rename_keys.append((f"transformer.decoder.layers.{i}.norm3.weight", f"decoder.layers.{i}.final_layer_norm.weight"))
    rename_keys.append((f"transformer.decoder.layers.{i}.norm3.bias", f"decoder.layers.{i}.final_layer_norm.bias"))

# convolutional projection + query embeddings + layernorm of decoder + class and bounding coord heads
rename_keys.extend(
    [
        ("input_proj.weight", "input_projection.weight"),
        ("input_proj.bias", "input_projection.bias"),
        ("query_embed.weight", "query_position_embeddings.weight"),
        ("transformer.decoder.norm.weight", "decoder.layernorm.weight"),
        ("transformer.decoder.norm.bias", "decoder.layernorm.bias"),
        ("class_embed.weight", "class_labels_classifier.weight"),
        ("class_embed.bias", "class_labels_classifier.bias"),
        ("kpt_embed.layers.0.weight", "coord_predictor.layers.0.weight"),
        ("kpt_embed.layers.0.bias", "coord_predictor.layers.0.bias"),
        ("kpt_embed.layers.1.weight", "coord_predictor.layers.1.weight"),
        ("kpt_embed.layers.1.bias", "coord_predictor.layers.1.bias"),
        ("kpt_embed.layers.2.weight", "coord_predictor.layers.2.weight"),
        ("kpt_embed.layers.2.bias", "coord_predictor.layers.2.bias"),
    ]
)


def rename_key(state_dict, old, new):
    val = state_dict.pop(old)
    state_dict[new] = val


def rename_backbone_keys(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if "backbone.0.body" in key:
            new_key = key.replace("backbone.0.body", "backbone.conv_encoder.body")
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict


def read_in_q_k_v(state_dict, is_panoptic=False):
    prefix = ""
    if is_panoptic:
        prefix = "prtr."

    # first: transformer encoder
    for i in range(6):
        # read in weights + bias of input projection layer (in PyTorch's MultiHeadAttention, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.encoder.layers.{i}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"encoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"encoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"encoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
    # next: transformer decoder (which is a bit more complex because it also includes cross-attention)
    for i in range(6):
        # read in weights + bias of input projection layer of self-attention
        in_proj_weight = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
        # read in weights + bias of input projection layer of cross-attention
        in_proj_weight_cross_attn = state_dict.pop(
            f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_weight"
        )
        in_proj_bias_cross_attn = state_dict.pop(f"{prefix}transformer.decoder.layers.{i}.multihead_attn.in_proj_bias")
        # next, add query, keys and values (in that order) of cross-attention to the state dict
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.weight"] = in_proj_weight_cross_attn[:256, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.q_proj.bias"] = in_proj_bias_cross_attn[:256]
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.weight"] = in_proj_weight_cross_attn[256:512, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.k_proj.bias"] = in_proj_bias_cross_attn[256:512]
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.weight"] = in_proj_weight_cross_attn[-256:, :]
        state_dict[f"decoder.layers.{i}.encoder_attn.v_proj.bias"] = in_proj_bias_cross_attn[-256:]


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    return im

def model_key_helper(state_dict):
    import re
    exp = re.compile(r'^backbone\.body')
    return {exp.sub('backbone.0.body', k): v for k, v in state_dict.items()}

@torch.no_grad()
def convert_prtr_checkpoint(model_name, pytorch_dump_folder_path):
    """
    Copy/paste/tweak model's weights to our PRTR structure.
    """
    prtrmodel = get_pose_net(cfg, is_train=False)
    weights = torch.load(model_name, map_location="cpu")
    prtrmodel.load_state_dict(model_key_helper(weights))
    
    # load default config
    mconfig = PrtrConfig()
    # set backbone and dilation attributes
    if "resnet101" in model_name:
        mconfig.backbone = "resnet101"
    if "dc5" in model_name:
        mconfig.dilation = True
    
    is_panoptic = False

    mconfig.num_labels = 17
    repo_id = "datasets/huggingface/label-files"
    filename = "coco-detection-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    

    # load feature extractor
    format = "coco_panoptic" if is_panoptic else "coco_detection"
    feature_extractor = PrtrFeatureExtractor(format=format)

    # prepare image
    img = prepare_img()
    encoding = feature_extractor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    logger.info(f"Converting model {model_name}...")

    # load original model from torch hub
    prtr = prtrmodel
    state_dict = prtr.state_dict()
    # rename keys
    for src, dest in rename_keys:
        if is_panoptic:
            src = "prtr." + src
        rename_key(state_dict, src, dest)
    state_dict = rename_backbone_keys(state_dict)
    # query, key and value matrices need special treatment
    read_in_q_k_v(state_dict, is_panoptic=is_panoptic)
    # important: we need to prepend a prefix to each of the base model keys as the head models use different attributes for them
    prefix = "prtr.model." if is_panoptic else "model."
    for key in state_dict.copy().keys():
        if is_panoptic:
            if (
                key.startswith("prtr")
                and not key.startswith("class_labels_classifier")
                and not key.startswith("coord_predictor")
            ):
                val = state_dict.pop(key)
                state_dict["prtr.model" + key[4:]] = val
            elif "class_labels_classifier" in key or "coord_predictor" in key:
                val = state_dict.pop(key)
                state_dict["prtr." + key] = val
            elif key.startswith("coord_attention") or key.startswith("mask_head"):
                continue
            else:
                val = state_dict.pop(key)
                state_dict[prefix + key] = val
        else:
            if not key.startswith("class_labels_classifier") and not key.startswith("coord_predictor"):
                val = state_dict.pop(key)
                state_dict[prefix + key] = val
    # finally, create HuggingFace model and load state dict
    
    print(mconfig.num_labels)
    model = PrtrForObjectDetection(mconfig)
    model.load_state_dict(state_dict)
    model.eval()
    # verify our conversion
    original_outputs = prtr(pixel_values)
    outputs = model(pixel_values)
    assert torch.allclose(outputs.logits, original_outputs["pred_logits"], atol=1e-4)
    assert torch.allclose(outputs.pred_coords, original_outputs["pred_coords"], atol=1e-4)
    
    # Save model and feature extractor
    logger.info(f"Saving PyTorch model and feature extractor to {pytorch_dump_folder_path}...")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)
    feature_extractor.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name", default="pose_resnet50.pth", type=str, help="Name of the PRTR model you'd like to convert."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    args = parser.parse_args()
    convert_prtr_checkpoint(args.model_name, args.pytorch_dump_folder_path)
