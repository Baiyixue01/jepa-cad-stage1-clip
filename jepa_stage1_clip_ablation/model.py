import torch
from torch import nn


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def _pool_vision_output(outputs) -> torch.Tensor:
    if getattr(outputs, "pooler_output", None) is not None:
        return outputs.pooler_output
    return outputs.last_hidden_state[:, 0]


class JEPADinoTextPredictor(nn.Module):
    def __init__(self, source_vision, target_vision, text_encoder, exp_cfg):
        super().__init__()
        self.source_vision = source_vision
        self.target_vision = target_vision
        self.text_encoder = text_encoder
        self.freeze_source_vision = exp_cfg["freeze_source_vision"]
        self.freeze_text_encoder = exp_cfg["freeze_text_encoder"]

        self.source_image_proj = nn.Sequential(
            nn.LayerNorm(exp_cfg["vision_embed_dim"]),
            nn.Linear(exp_cfg["vision_embed_dim"], exp_cfg["fusion_dim"]),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(exp_cfg["text_embed_dim"]),
            nn.Linear(exp_cfg["text_embed_dim"], exp_cfg["fusion_dim"]),
        )
        self.fusion = nn.Sequential(
            nn.Linear(exp_cfg["fusion_dim"] * 2, exp_cfg["fusion_dim"]),
            nn.GELU(),
            nn.LayerNorm(exp_cfg["fusion_dim"]),
            nn.Linear(exp_cfg["fusion_dim"], exp_cfg["fusion_dim"]),
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(exp_cfg["fusion_dim"]),
            nn.Linear(exp_cfg["fusion_dim"], exp_cfg["fusion_dim"] * 4),
            nn.GELU(),
            nn.Linear(exp_cfg["fusion_dim"] * 4, exp_cfg["target_embed_dim"]),
        )

    def encode_source_image(self, pixel_values):
        if self.freeze_source_vision:
            with torch.no_grad():
                outputs = self.source_vision(pixel_values=pixel_values)
        else:
            outputs = self.source_vision(pixel_values=pixel_values)
        return _pool_vision_output(outputs)

    def encode_text(self, input_ids, attention_mask):
        if self.freeze_text_encoder:
            with torch.no_grad():
                outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        if getattr(outputs, "text_embeds", None) is not None:
            return outputs.text_embeds
        return outputs.pooler_output

    def encode_target_image(self, pixel_values):
        with torch.no_grad():
            outputs = self.target_vision(pixel_values=pixel_values)
        return _pool_vision_output(outputs)

    def forward(self, before_pixel_values, input_ids, attention_mask, highlight_pixel_values):
        z_before = self.encode_source_image(before_pixel_values)
        z_text = self.encode_text(input_ids, attention_mask)
        z_before = self.source_image_proj(z_before)
        z_text = self.text_proj(z_text)
        z_context = self.fusion(torch.cat([z_before, z_text], dim=-1))
        z_pred = self.predictor(z_context)
        z_target = self.encode_target_image(highlight_pixel_values)
        return {
            "z_pred": z_pred,
            "z_target": z_target,
        }


def _apply_lora_to_source_vision(source_vision, exp_cfg):
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=exp_cfg["lora_rank"],
        lora_alpha=exp_cfg["lora_alpha"],
        target_modules=exp_cfg["lora_target_modules"],
        lora_dropout=exp_cfg["lora_dropout"],
        bias="none",
    )
    return get_peft_model(source_vision, lora_config)


def create_model(exp_cfg):
    from transformers import AutoModel, CLIPTextModelWithProjection

    source_vision = AutoModel.from_pretrained(exp_cfg["vision_model_name"])
    if exp_cfg["train_source_lora"]:
        source_vision = _apply_lora_to_source_vision(source_vision, exp_cfg)
    elif exp_cfg["freeze_source_vision"]:
        freeze_module(source_vision)

    target_vision = AutoModel.from_pretrained(exp_cfg["target_vision_model_name"])
    freeze_module(target_vision)
    target_vision.eval()

    text_encoder = CLIPTextModelWithProjection.from_pretrained(exp_cfg["text_model_name"])
    if exp_cfg["freeze_text_encoder"]:
        freeze_module(text_encoder)
    text_encoder.eval()

    model = JEPADinoTextPredictor(
        source_vision=source_vision,
        target_vision=target_vision,
        text_encoder=text_encoder,
        exp_cfg=exp_cfg,
    )
    return model
