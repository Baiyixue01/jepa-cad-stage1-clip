from copy import deepcopy

import torch
from torch import nn

from . import config


class JEPAClipPredictor(nn.Module):
    def __init__(self, clip_model, target_clip_model=None):
        super().__init__()
        self.clip = clip_model
        self.target_clip = target_clip_model or deepcopy(clip_model)
        self.fusion = nn.Sequential(
            nn.Linear(config.EMBED_DIM * 2, config.EMBED_DIM),
            nn.GELU(),
            nn.LayerNorm(config.EMBED_DIM),
            nn.Linear(config.EMBED_DIM, config.EMBED_DIM),
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(config.EMBED_DIM),
            nn.Linear(config.EMBED_DIM, config.EMBED_DIM * 4),
            nn.GELU(),
            nn.Linear(config.EMBED_DIM * 4, config.EMBED_DIM),
        )

    def forward(self, before_pixel_values, input_ids, attention_mask, highlight_pixel_values):
        z_before = self.clip.get_image_features(pixel_values=before_pixel_values)
        z_text = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        z_context = self.fusion(torch.cat([z_before, z_text], dim=-1))
        z_pred = self.predictor(z_context)

        with torch.no_grad():
            z_target = self.target_clip.get_image_features(pixel_values=highlight_pixel_values)

        return {
            "z_pred": z_pred,
            "z_target": z_target,
        }


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def create_model(exp_cfg):
    from transformers import CLIPModel

    clip = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
    if exp_cfg["use_lora"]:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=exp_cfg["lora_rank"],
            lora_alpha=exp_cfg["lora_alpha"],
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=exp_cfg["lora_dropout"],
            bias="none",
        )
        clip = get_peft_model(clip, lora_config)

    target_clip = CLIPModel.from_pretrained(config.CLIP_MODEL_NAME)
    freeze_module(target_clip)
    target_clip.eval()

    model = JEPAClipPredictor(clip_model=clip, target_clip_model=target_clip)

    if exp_cfg["freeze_clip_image_encoder"] and hasattr(model.clip, "vision_model"):
        freeze_module(model.clip.vision_model)
        if hasattr(model.clip, "visual_projection"):
            freeze_module(model.clip.visual_projection)
    if exp_cfg["freeze_clip_text_encoder"] and hasattr(model.clip, "text_model"):
        freeze_module(model.clip.text_model)
        if hasattr(model.clip, "text_projection"):
            freeze_module(model.clip.text_projection)

    return model
