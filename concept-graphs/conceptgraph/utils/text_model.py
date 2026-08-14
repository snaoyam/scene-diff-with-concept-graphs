# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from ultralytics.utils import WEIGHTS_DIR, checks
from ultralytics.utils.torch_utils import smart_inference_mode

try:
    import open_clip
except ImportError:
    checks.check_requirements("open_clip_torch")
    import open_clip


class TextModel(nn.Module):
    """Abstract base class for text encoding models.

    This class defines the interface for text encoding models used in vision-language tasks. Subclasses must implement
    the tokenize and encode_text methods to provide text tokenization and encoding functionality.

    Methods:
        tokenize: Convert input texts to tokens for model processing.
        encode_text: Encode tokenized texts into normalized feature vectors.
    """

    def __init__(self):
        """Initialize the TextModel base class."""
        super().__init__()

    @abstractmethod
    def tokenize(self, texts):
        """Convert input texts to tokens for model processing."""
        pass

    @abstractmethod
    def encode_text(self, texts, dtype):
        """Encode tokenized texts into normalized feature vectors."""
        pass


class CLIP(TextModel):
    """Implements OpenAI's CLIP (Contrastive Language-Image Pre-training) text encoder using open_clip.

    This class provides a text encoder based on the CLIP model, which can convert text into feature vectors that
    are aligned with corresponding image features in a shared embedding space.

    Attributes:
        model: The loaded open_clip model.
        image_preprocess (callable): Preprocessing transform for images (validation).
        device (torch.device): Device where the model is loaded.
        tokenizer: Tokenizer instance for the specific model.

    Methods:
        tokenize: Convert input texts to CLIP tokens.
        encode_text: Encode tokenized texts into normalized feature vectors.
    """

    def __init__(self, size: str, device: torch.device) -> None:
        """Initialize the CLIP text encoder using open_clip.

        This class implements the TextModel interface using open_clip for text encoding. It loads a
        pre-trained CLIP model of the specified size and prepares it for text encoding tasks.

        Args:
            size (str): Model size identifier (e.g., 'ViT-B/32').
            device (torch.device): Device to load the model on.
        """
        super().__init__()
        
        # open_clip은 'ViT-B/32' 대신 'ViT-B-32' 형식을 사용합니다.
        model_name = size.replace("/", "-")
        
        # open_clip.create_model_and_transforms는 (model, preprocess_train, preprocess_val)을 반환합니다.
        self.model, _, self.image_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained="openai", device=device, cache_dir=str(WEIGHTS_DIR / "clip")
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts: str | list[str], truncate: bool = True) -> torch.Tensor:
        """Convert input texts to CLIP tokens using open_clip.

        Args:
            texts (str | list[str]): Input text or list of texts to tokenize.
            truncate (bool, optional): Note that open_clip tokenizer might handle context limits natively.
                This argument is preserved for API compatibility.

        Returns:
            (torch.Tensor): Tokenized text tensor with shape (batch_size, context_length) ready for model processing.
        """
        # open_clip의 tokenizer는 함수 형태이며 직접 texts를 입력받습니다.
        return self.tokenizer(texts).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Encode tokenized texts into normalized feature vectors."""
        txt_feats = self.model.encode_text(texts).to(dtype)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        return txt_feats

    @smart_inference_mode()
    def encode_image(self, image: Image.Image | torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Encode images into normalized feature vectors."""
        if isinstance(image, Image.Image):
            image = self.image_preprocess(image).unsqueeze(0).to(self.device)
        img_feats = self.model.encode_image(image).to(dtype)
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        return img_feats


class MobileCLIP(TextModel):
    """Implement Apple's MobileCLIP text encoder for efficient text encoding."""

    config_size_map = {"s0": "s0", "s1": "s1", "s2": "s2", "b": "b", "blt": "b"}

    def __init__(self, size: str, device: torch.device) -> None:
        """Initialize the MobileCLIP text encoder."""
        try:
            import mobileclip
        except ImportError:
            # Ultralytics fork preferred since Apple MobileCLIP repo has incorrect version of torchvision
            checks.check_requirements("git+https://github.com/ultralytics/mobileclip.git")
            import mobileclip

        super().__init__()
        config = self.config_size_map[size]
        file = f"mobileclip_{size}.pt"
        if not Path(file).is_file():
            from ultralytics import download

            download(f"https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/{file}")
        self.model = mobileclip.create_model_and_transforms(f"mobileclip_{config}", pretrained=file, device=device)[0]
        self.tokenizer = mobileclip.get_tokenizer(f"mobileclip_{config}")
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts: list[str]) -> torch.Tensor:
        """Convert input texts to MobileCLIP tokens."""
        return self.tokenizer(texts).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Encode tokenized texts into normalized feature vectors."""
        text_features = self.model.encode_text(texts).to(dtype)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features


class MobileCLIPTS(TextModel):
    """Load a TorchScript traced version of MobileCLIP."""

    def __init__(self, device: torch.device, weight: str = "mobileclip_blt.ts"):
        """Initialize the MobileCLIP TorchScript text encoder."""
        super().__init__()
        from ultralytics.utils.downloads import attempt_download_asset

        self.encoder = torch.jit.load(attempt_download_asset(weight), map_location=device)
        # clip.clip.tokenize 대신 open_clip의 tokenize 사용
        self.tokenizer = open_clip.tokenize
        self.device = device

    def tokenize(self, texts: list[str], truncate: bool = True) -> torch.Tensor:
        """Convert input texts to MobileCLIP tokens."""
        # open_clip.tokenize는 기본적으로 context length에 맞게 조정됩니다.
        return self.tokenizer(texts).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Encode tokenized texts into normalized feature vectors."""
        # NOTE: no need to do normalization here as it's embedded in the torchscript model
        return self.encoder(texts).to(dtype)


def build_text_model(variant: str, device: torch.device = None) -> TextModel:
    """Build a text encoding model based on the specified variant."""
    base, size = variant.split(":")
    if base == "clip":
        return CLIP(size, device)
    elif base == "mobileclip":
        return MobileCLIPTS(device)
    elif base == "mobileclip2":
        return MobileCLIPTS(device, weight="mobileclip2_b.ts")
    else:
        raise ValueError(f"Unrecognized base model '{base}'. Supported models are 'clip', 'mobileclip', 'mobileclip2'.")