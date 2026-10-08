"""
src/embeddings/clip_encoder.py — OpenCLIP-based image and text embedding.

Designed for 4GB VRAM: uses ViT-B/32, processes in batches,
unloads from GPU after use if memory pressure is detected.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image

from src.config import cfg

logger = logging.getLogger(__name__)


class CLIPEncoder:
    """
    OpenCLIP-based encoder for image and text embeddings.

    Produces 512-dim normalized embeddings (ViT-B/32).
    Uses FP16 on CUDA for memory efficiency.
    """

    def __init__(self,
                 model_name: str = None,
                 pretrained: str = None,
                 device: str = None):
        self.model_name = model_name or cfg.clip_model
        self.pretrained = pretrained or cfg.clip_pretrained
        self.device = device or cfg.device
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load CLIP model into memory."""
        if self._loaded:
            return

        try:
            import open_clip
            import torch

            logger.info("Loading CLIP: %s / %s on %s",
                        self.model_name, self.pretrained, self.device)

            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device,
            )
            self._tokenizer = open_clip.get_tokenizer(self.model_name)

            if self.device.startswith("cuda") and cfg.use_fp16:
                self._model = self._model.half()
                logger.info("CLIP: FP16 enabled")

            self._model.eval()
            self._loaded = True
            logger.info("CLIP model loaded successfully")

        except ImportError:
            logger.error("open_clip_torch not installed. Run: pip install open-clip-torch")
            raise
        except Exception as e:
            logger.error("Failed to load CLIP: %s", e)
            raise

    def unload(self) -> None:
        """Release CLIP from GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._preprocess = None
            self._tokenizer = None
            self._loaded = False
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("CLIP model unloaded")

    def encode_images(self, images: List[Union[np.ndarray, Image.Image, str]],
                      batch_size: int = None) -> np.ndarray:
        """
        Encode a list of images to normalized embeddings.

        Args:
            images: List of BGR numpy arrays, PIL Images, or file paths.
            batch_size: Override config batch size.

        Returns:
            np.ndarray of shape (N, embedding_dim), float32, L2-normalized.
        """
        if not self._loaded:
            self.load()

        import torch

        batch_size = batch_size or cfg.clip_batch_size
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            pil_images = []

            for img in batch:
                if isinstance(img, str):
                    if Path(img).exists():
                        pil_images.append(Image.open(img).convert("RGB"))
                    else:
                        # Use blank image as placeholder
                        pil_images.append(Image.new("RGB", (224, 224)))
                elif isinstance(img, np.ndarray):
                    import cv2
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_images.append(Image.fromarray(rgb))
                elif isinstance(img, Image.Image):
                    pil_images.append(img.convert("RGB"))
                else:
                    pil_images.append(Image.new("RGB", (224, 224)))

            # Preprocess and stack
            tensors = torch.stack([self._preprocess(img) for img in pil_images])

            if self.device.startswith("cuda") and cfg.use_fp16:
                tensors = tensors.half()

            tensors = tensors.to(self.device)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.startswith("cuda")):
                features = self._model.encode_image(tensors)
                features = features / features.norm(dim=-1, keepdim=True)

            all_embeddings.append(features.cpu().float().numpy())

        return np.concatenate(all_embeddings, axis=0) if all_embeddings else np.zeros((0, cfg.embedding_dim))

    def encode_text(self, texts: List[str]) -> np.ndarray:
        """
        Encode text queries to normalized embeddings.

        Returns:
            np.ndarray of shape (N, embedding_dim), float32, L2-normalized.
        """
        if not self._loaded:
            self.load()

        import torch

        tokens = self._tokenizer(texts).to(self.device)

        with torch.no_grad():
            features = self._model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().float().numpy()

    def encode_single_image(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """Convenience method for a single image. Returns (embedding_dim,) array."""
        result = self.encode_images([image])
        return result[0] if len(result) > 0 else np.zeros(cfg.embedding_dim)

    def encode_single_text(self, text: str) -> np.ndarray:
        """Convenience method for a single text. Returns (embedding_dim,) array."""
        result = self.encode_text([text])
        return result[0] if len(result) > 0 else np.zeros(cfg.embedding_dim)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *args):
        self.unload()

    # Convenience aliases
    encode_image = encode_single_image


# Alias for convenience
ClipEncoder = CLIPEncoder
