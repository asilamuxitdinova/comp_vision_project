"""Build a pretrained timm model with a new classification head."""
import timm
import torch.nn as nn


def build_model(cfg: dict, num_classes: int, pretrained: bool | None = None) -> nn.Module:
    m = cfg["model"]
    kwargs = dict(
        pretrained=m.get("pretrained", True) if pretrained is None else pretrained,
        num_classes=num_classes,
        drop_rate=m.get("drop_rate", 0.0),
        drop_path_rate=m.get("drop_path_rate", 0.0),
    )
    if m["name"].startswith(("vit_", "deit", "eva", "beit")):
        kwargs["img_size"] = cfg["data"]["img_size"]  # ViTs need the input size up front
    return timm.create_model(m["name"], **kwargs)


def normalization(model: nn.Module) -> tuple[tuple, tuple]:
    """Mean/std the pretrained weights expect."""
    dc = timm.data.resolve_model_data_config(model)
    return tuple(dc["mean"]), tuple(dc["std"])
