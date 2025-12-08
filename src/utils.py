"""
Utility functions for model loading, feature extraction, and data processing.

This module provides:
    - Model and transform loading for various architectures (UNI, Virchow, Phikon, etc.)
    - Feature extraction from pretrained models
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import timm
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
from transformers import AutoModel, ViTModel


@torch.no_grad()
def extract_patch_features_from_dataloader(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    forward_fn: Callable,
) -> Dict[str, np.ndarray]:
    """Extract features and labels from images using a pretrained model.

    Adapted from the UNI library for feature extraction from histopathology patches.

    Args:
        model (torch.nn.Module): Pretrained model for feature extraction.
        dataloader (torch.utils.data.DataLoader): DataLoader yielding image batches.
        forward_fn (Callable): Function to extract embeddings from the model.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing:
            - 'embeddings': [N x D] array of feature embeddings
            - 'labels': [N] array of labels
            - 'files': List of file paths
            - 'coords': [N x 2] array of patch coordinates
    """
    all_embeddings = []
    all_labels = []
    all_files = []
    all_coords = []

    batch_size = dataloader.batch_size
    device = next(model.parameters()).device

    for batch_idx, (batch, target, files, coords) in tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Extracting features"
    ):
        remaining = batch.shape[0]

        # Pad batch if incomplete
        if remaining != batch_size:
            padding = torch.zeros((batch_size - remaining,) + batch.shape[1:]).type(
                batch.type()
            )
            batch = torch.vstack([batch, padding])

        batch = batch.to(device)

        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            embeddings = forward_fn(model, batch, remaining)

            # Handle detection datasets with bounding box labels
            if isinstance(target, list):
                labels = np.array(
                    [1 if t["labels"].sum() > 0 else 0 for t in target]
                )[:remaining]
            else:
                labels = target.numpy()[:remaining]

            assert not torch.isnan(embeddings).any(), "NaN detected in embeddings"

        all_embeddings.append(embeddings)
        all_labels.append(labels)
        all_files.extend(files)
        all_coords.append(coords)

    asset_dict = {
        "embeddings": np.vstack(all_embeddings).astype(np.float32),
        "labels": np.concatenate(all_labels),
        "files": all_files,
        "coords": np.vstack(all_coords),
    }

    return asset_dict


def return_forward(model_name: str) -> Callable:
    """Return the appropriate forward function for feature extraction.

    Args:
        model_name (str): Name of the model architecture.

    Returns:
        Callable: Forward function that takes (model, batch, remaining) and returns embeddings.

    Raises:
        NotImplementedError: If the model is not supported.
    """

    def forward_uni(model, batch, remaining):
        """Forward pass for UNI model."""
        embeddings = model(batch).detach().cpu()[:remaining, :]
        return embeddings

    def forward_virchow(model, batch, remaining):
        """Forward pass for Virchow/Virchow2 models (class + patch tokens)."""
        output = model(batch).detach().cpu()[:remaining, :]
        class_token = output[:, 0]
        patch_tokens = output[:, 1:]
        embeddings = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)
        return embeddings

    def forward_vit_cls(model, batch, remaining):
        """Forward pass for ViT models using [CLS] token."""
        embeddings = (
            model(batch).last_hidden_state[:, 0, :].detach().cpu()[:remaining, :]
        )
        return embeddings

    def forward_standard(model, batch, remaining):
        """Forward pass for standard models (ResNet, GigaPath, etc.)."""
        embeddings = model(batch).detach().cpu()[:remaining, :]
        return embeddings

    def forward_vit_h(model, batch, remaining):
        """Forward pass for ViT-H (class + mean patch tokens)."""
        output = model(batch).last_hidden_state.detach().cpu()[:remaining, :]
        class_token = output[:, 0]
        patch_tokens = output[:, 1:]
        embeddings = torch.cat([class_token, patch_tokens.mean(dim=1)], dim=-1)
        return embeddings

    # Map model names to forward functions
    forward_map = {
        "uni": forward_uni,
        "virchow": forward_virchow,
        "virchow2": forward_virchow,
        "phikon": forward_vit_cls,
        "ViT_S_DINOv3": forward_vit_cls,
        "resnet50": forward_standard,
        "gigapath": forward_standard,
        "hoptimus": forward_standard,
        "ViT_H": forward_vit_h,
    }

    if model_name not in forward_map:
        raise NotImplementedError(f"Model '{model_name}' not implemented")

    return forward_map[model_name]


def load_model_and_transforms(model_name: str) -> Tuple[torch.nn.Module, T.Compose]:
    """Load a pretrained model and its corresponding transforms.

    Args:
        model_name (str): Name of the model to load. Supported models include:
            - 'uni': UNI foundation model
            - 'virchow', 'virchow2': Virchow foundation models
            - 'phikon': Phikon ViT model
            - 'gigapath': GigaPath foundation model
            - 'hoptimus': H-optimus foundation model
            - 'resnet50': ResNet-50 pretrained on ImageNet
            - 'ViT_H', 'ViT_S', 'ViT_tiny': Vision Transformers
            - 'ViT_S_DINOv3': DINOv3 small ViT
            - 'efficientnet_b0', 'efficientnet_b3', 'efficientnet_b7': EfficientNets
            - 'densenet_121': DenseNet-121
            - 'Swin_base': Swin Transformer
            - 'convnext_base': ConvNeXt

    Returns:
        Tuple[torch.nn.Module, T.Compose]: Model and transforms.

    Raises:
        ValueError: If the model name is not recognized.
        NotImplementedError: If the model is not yet implemented.
    """
    if model_name == "uni":
        model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        model.load_state_dict(torch.load("UNI/checkpoints/UNI/pytorch_model.bin"))
        transforms = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    elif model_name == "ViT_tiny":
        model = timm.create_model(
            "vit_tiny_patch16_224.augreg_in21k", pretrained=True, num_classes=1
        )
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

    elif model_name == "ViT_S":
        model = timm.create_model(
            "vit_small_patch16_224.augreg_in21k", pretrained=True, num_classes=1
        )
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

    elif model_name == "Swin_base":
        from transformers import SwinForImageClassification

        model = SwinForImageClassification.from_pretrained(
            "microsoft/swin-base-patch4-window7-224",
            num_labels=1,
            ignore_mismatched_sizes=True,
        )
        transforms = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    elif model_name == "efficientnet_b0":
        model = timm.create_model(
            "efficientnet_b0.ra_in1k", pretrained=True, num_classes=1
        )
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

    elif model_name == "efficientnet_b3":
        model = timm.create_model(
            "efficientnet_b3.ra2_in1k", pretrained=True, num_classes=1
        )
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

    elif model_name == "efficientnet_b7":
        model = timm.create_model(
            "tf_efficientnet_b7.ra_in1k", pretrained=True, num_classes=1
        )
        transforms = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    elif model_name == "densenet_121":
        model = timm.create_model(
            "densenet121.ra_in1k", pretrained=True, num_classes=1
        )
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)

    elif model_name == "convnext_base":
        from transformers import ConvNextForImageClassification

        model = ConvNextForImageClassification.from_pretrained(
            "facebook/convnext-base-224-22k", num_labels=1
        )
        transforms = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    elif model_name == "virchow":
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow",
            pretrained=False,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        model.load_state_dict(torch.load("checkpoints/Virchow/pytorch_model.bin"))
        transforms = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )

    elif model_name == "virchow2":
        model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=False,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        model.load_state_dict(torch.load("checkpoints/Virchow2/pytorch_model.bin"))
        transforms = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )

    elif model_name == "phikon":
        model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        transforms = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    elif model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()
        transforms = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    elif model_name == "ViT_H":
        model = ViTModel.from_pretrained("google/vit-huge-patch14-224-in21k")
        transforms = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    elif model_name == "ViT_S_DINOv3":
        model = AutoModel.from_pretrained(
            "facebook/dinov3-vits16-pretrain-lvd1689m", device_map="auto"
        )
        transforms = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    elif model_name == "gigapath":
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        transforms = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    elif model_name == "hoptimus":
        model = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        transforms = T.Compose([
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(
                mean=(0.707223, 0.578729, 0.703617),
                std=(0.211883, 0.230117, 0.177517),
            ),
        ])

    elif model_name == "ctranspath":
        raise NotImplementedError("CTransPath model is not yet implemented")

    else:
        raise ValueError(f"Model '{model_name}' not implemented")

    return model, transforms


def collate_fn(
    batch: List[Tuple[torch.Tensor, int, str, Tuple[int, int]]]
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[Tuple[int, int]]]:
    """Custom collate function for DataLoader.

    Args:
        batch (List[Tuple]): List of tuples containing (image, label, file, coords).

    Returns:
        Tuple: Batched images, labels, files, and coordinates.
    """
    images = []
    targets = []
    files = []
    coords = []

    for image, target, file, coord in batch:
        images.append(image)
        targets.append(target)
        files.append(file)
        coords.append(coord)

    images = torch.stack(images, dim=0)
    targets = torch.tensor(targets, dtype=torch.int64)
    coords = torch.stack([torch.tensor(c) for c in coords], dim=0)

    return images, targets, files, coords