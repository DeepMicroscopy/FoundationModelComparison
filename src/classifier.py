"""
Classifier module for histopathology image classification.

This module provides:
    - Unified Classifier interface supporting multiple architectures
    - LoRA (Low-Rank Adaptation) support for efficient fine-tuning
"""

from __future__ import annotations

import os
from typing import Tuple

import peft
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import Compose
from transformers import AutoModel, ViTModel

from .utils import load_model_and_transforms

# LoRA target modules for each supported architecture
VALID_LORA_MODULES = {
    "hoptimus": ["qkv", "proj", "fc1", "fc2"],
    "virchow2": ["qkv", "proj", "fc1", "fc2"],
    "virchow": ["qkv", "proj", "fc1", "fc2"],
    "ViT_H": ["query", "key", "value", "dense"],
    "ViT_S_DINOv3": ["k_proj", "v_proj", "q_proj", "o_proj", "up_proj", "down_proj"],
    "uni": ["qkv", "proj", "fc1", "fc2"],
    "gigapath": ["qkv", "proj", "fc1", "fc2"],
    "phikon": ["query", "key", "value", "dense"],
}

# Models requiring custom classification heads
VIT_FOR_CLASSIFICATION = {
    "ViT_H",
    "ViT_B",
    "ViT_L",
    "phikon",
    "ViT_S_DINOv3",
}


class ViTForClassification(nn.Module):
    """Vision Transformer (ViT) wrapper with custom classification head.

    This class wraps pretrained ViT models and adds a linear classification head
    on top of the [CLS] token output.

    Args:
        model_name (str): Name of the ViT model to load.
        num_classes (int, optional): Number of output classes. Defaults to 2.
            For binary classification, outputs 1 logit.
    """

    def __init__(self, model_name: str, num_classes: int = 2):
        super().__init__()
        self.model_name = model_name
        self.vit = self._load_backbone(model_name)

        # Binary classification uses single output
        num_outputs = 1 if num_classes == 2 else num_classes

        hidden_size = self._get_hidden_size(model_name)
        if hidden_size is None:
            raise ValueError(
                f"Unknown model: {model_name}. Please update _get_hidden_size()."
            )

        self.head = nn.Linear(hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ViT and classification head.

        Args:
            x (torch.Tensor): Input images with shape [B, 3, H, W].

        Returns:
            torch.Tensor: Logits with shape [B, num_classes].
        """
        outputs = self.vit(x)
        cls_token = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token
        logits = self.head(cls_token)
        return logits

    @staticmethod
    def _get_hidden_size(model_name: str) -> int:
        """Get the hidden size for a given ViT model.

        Args:
            model_name (str): Name of the ViT model.

        Returns:
            int: Hidden size of the model, or None if unknown.
        """
        hidden_sizes = {
            "phikon": 768,
            "ViT_H": 1280,
            "ViT_B": 768,
            "ViT_L": 1024,
            "ViT_S_DINOv3": 384,
        }
        return hidden_sizes.get(model_name)

    @staticmethod
    def _load_backbone(model_name: str) -> nn.Module:
        """Load the pretrained ViT backbone.

        Args:
            model_name (str): Name of the ViT model.

        Returns:
            nn.Module: Loaded ViT model.

        Raises:
            ValueError: If the model is not implemented.
        """
        if model_name == "phikon":
            return ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
        elif model_name == "ViT_H":
            return ViTModel.from_pretrained("google/vit-huge-patch14-224-in21k")
        elif model_name == "ViT_B":
            return ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        elif model_name == "ViT_L":
            return ViTModel.from_pretrained("google/vit-large-patch16-224-in21k")
        elif model_name == "ViT_S_DINOv3":
            return AutoModel.from_pretrained(
                "facebook/dinov3-vits16-pretrain-lvd1689m", device_map="cpu"
            )
        else:
            raise ValueError(f"Model '{model_name}' not implemented")

    @staticmethod
    def get_transforms(model_name: str) -> Compose:
        """Get the input transforms for a given ViT model.

        Args:
            model_name (str): Name of the ViT model.

        Returns:
            Compose: Composed transforms for preprocessing.

        Raises:
            ValueError: If transforms are not defined for the model.
        """
        if model_name in ["phikon", "ViT_H", "ViT_B", "ViT_L", "ViT_S_DINOv3"]:
            return T.Compose([
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            raise ValueError(f"Transforms not implemented for model: {model_name}")


class Classifier(nn.Module):
    """Unified classifier supporting multiple architectures and LoRA fine-tuning.

    This class provides a unified interface for loading and using various pretrained
    models for histopathology image classification, including ResNet, ViT variants,
    and foundation models like UNI, Virchow, and GigaPath.

    Args:
        model_name (str, optional): Name of the model architecture. Defaults to 'resnet50'.
        lora (bool, optional): Whether to apply LoRA adaptation. Defaults to False.
        num_classes (int, optional): Number of output classes. Defaults to 2.

    Attributes:
        model (nn.Module): The loaded model.
        input_transform (Compose): Transforms to apply to input images.
    """

    def __init__(
        self, model_name: str = "resnet50", lora: bool = False, num_classes: int = 2
    ):
        super().__init__()
        self.model_name = model_name
        self.lora = lora
        self.num_classes = num_classes
        self.model, self.input_transform = self._load_model(
            model_name, lora, num_classes
        )

    def _load_model(
        self, model_name: str, lora: bool = False, num_classes: int = 2
    ) -> Tuple[nn.Module, Compose]:
        """Load the specified model and its transforms.

        Args:
            model_name (str): Name of the model to load.
            lora (bool, optional): Whether to apply LoRA. Defaults to False.
            num_classes (int, optional): Number of classes. Defaults to 2.

        Returns:
            Tuple[nn.Module, Compose]: Loaded model and input transforms.

        Raises:
            ValueError: If the model is not implemented.
        """
        # Map model names to loading functions
        model_map = {
            "ViT_H": lambda: (
                ViTForClassification(model_name="ViT_H", num_classes=num_classes),
                ViTForClassification.get_transforms("ViT_H"),
            ),
            "ViT_B": lambda: (
                ViTForClassification(model_name="ViT_B", num_classes=num_classes),
                ViTForClassification.get_transforms("ViT_B"),
            ),
            "ViT_L": lambda: (
                ViTForClassification(model_name="ViT_L", num_classes=num_classes),
                ViTForClassification.get_transforms("ViT_L"),
            ),
            "ViT_S_DINOv3": lambda: (
                ViTForClassification(
                    model_name="ViT_S_DINOv3", num_classes=num_classes
                ),
                ViTForClassification.get_transforms("ViT_S_DINOv3"),
            ),
            "phikon": lambda: (
                ViTForClassification(model_name="phikon", num_classes=num_classes),
                ViTForClassification.get_transforms("phikon"),
            ),
            "ViT_tiny": lambda: load_model_and_transforms("ViT_tiny"),
            "ViT_S": lambda: load_model_and_transforms("ViT_S"),
            "efficientnet_b0": lambda: load_model_and_transforms("efficientnet_b0"),
            "efficientnet_b3": lambda: load_model_and_transforms("efficientnet_b3"),
            "efficientnet_b7": lambda: load_model_and_transforms("efficientnet_b7"),
            "densenet_121": lambda: load_model_and_transforms("densenet_121"),
            "Swin_base": lambda: load_model_and_transforms("Swin_base"),
            "convnext_base": lambda: load_model_and_transforms("convnext_base"),
            "resnet50": lambda: load_model_and_transforms("resnet50"),
            "hoptimus": lambda: load_model_and_transforms("hoptimus"),
            "virchow": lambda: load_model_and_transforms("virchow"),
            "virchow2": lambda: load_model_and_transforms("virchow2"),
            "uni": lambda: load_model_and_transforms("uni"),
            "gigapath": lambda: load_model_and_transforms("gigapath"),
        }

        if model_name not in model_map:
            raise ValueError(f"Model '{model_name}' not implemented")

        model, input_transforms = model_map[model_name]()

        # Add classification head if needed
        if model_name not in VIT_FOR_CLASSIFICATION:
            model = self._initialize_classification_head(model_name, model, num_classes)

        # Apply LoRA if requested
        if lora:
            model = self._initialize_lora_model(model_name, model)

        return model, input_transforms

    def _initialize_lora_model(
        self, model_name: str, model: nn.Module
    ) -> nn.Module:
        """Apply LoRA (Low-Rank Adaptation) to the model.

        Args:
            model_name (str): Name of the model.
            model (nn.Module): Base model to apply LoRA to.

        Returns:
            nn.Module: Model with LoRA applied.

        Raises:
            ValueError: If LoRA is not supported for the model.
        """
        if model_name not in VALID_LORA_MODULES:
            raise ValueError(f"No LoRA configuration available for '{model_name}'")

        config = peft.LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=VALID_LORA_MODULES[model_name],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["head"],
        )

        model = peft.get_peft_model(model, config)
        model.print_trainable_parameters()
        return model

    def _initialize_classification_head(
        self, model_name: str, model: nn.Module, num_classes: int = 2
    ) -> nn.Module:
        """Initialize or replace the classification head.

        Args:
            model_name (str): Name of the model.
            model (nn.Module): Model to modify.
            num_classes (int, optional): Number of output classes. Defaults to 2.

        Returns:
            nn.Module: Model with updated classification head.

        Raises:
            ValueError: If head initialization is not defined for the model.
        """
        num_outputs = 1 if num_classes == 2 else num_classes

        head_configs = {
            "resnet50": ("fc", 2048),
            "hoptimus": ("head", 1536),
            "virchow": ("head", 1280),
            "virchow2": ("head", 1280),
            "uni": ("head", 1024),
            "gigapath": ("head", 1536),
        }

        if model_name in head_configs:
            attr_name, in_features = head_configs[model_name]
            setattr(model, attr_name, nn.Linear(in_features, num_outputs))
        elif model_name in [
            "ViT_tiny",
            "ViT_S",
            "Swin_base",
            "convnext_base",
        ] or "efficientnet" in model_name or "densenet" in model_name:
            # These models already have classification heads
            pass
        else:
            raise ValueError(
                f"Classification head initialization not implemented for '{model_name}'"
            )

        return model

    def load_pretrained_model(self, checkpoint_path: str) -> nn.Module:
        """Load model weights from a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file (or directory for LoRA).

        Returns:
            nn.Module: Model with loaded weights.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
            RuntimeError: If loading fails.
        """
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            if self.lora:
                model = self.load_pretrained_lora_model(self.model_name, checkpoint_path)
            else:
                model = self.model
                model.load_state_dict(
                    torch.load(checkpoint_path, map_location="cpu")
                )
            return model
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from {checkpoint_path}: {str(e)}"
            )

    def load_pretrained_lora_model(
        self, model_name: str, checkpoint_dir: str
    ) -> nn.Module:
        """Load a LoRA-adapted model from a directory.

        Args:
            model_name (str): Name of the base model.
            checkpoint_dir (str): Directory containing LoRA weights.

        Returns:
            nn.Module: Model with LoRA weights loaded.
        """
        base_model, _ = self._load_model(model_name, lora=False, num_classes=self.num_classes)
        model = peft.PeftModel.from_pretrained(base_model, checkpoint_dir)
        return model

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the classifier.

        Args:
            x (torch.Tensor): Input images with shape [B, 3, H, W].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - logits: Raw model outputs
                - Y_prob: Sigmoid probabilities
                - Y_hat: Binary predictions (threshold 0.5)
        """
        output = self.model(x)

        # Handle different output formats
        if self.model_name in ["virchow", "virchow2"]:
            # Virchow models output [B, num_tokens, D], use [CLS] token
            logits = output[:, 0, :]
        elif self.model_name == "Swin_base":
            logits = output.logits
        else:
            logits = output

        logits = logits.squeeze()
        Y_prob = torch.sigmoid(logits)
        Y_hat = (Y_prob > 0.5).float()

        return logits, Y_prob, Y_hat