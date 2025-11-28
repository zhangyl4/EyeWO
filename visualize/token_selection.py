import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.vision_transformer import VisionTransformer, ViT_B_16_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class DenseViT(VisionTransformer):
    """
    Vision Transformer variant that exposes intermediate patch tokens and
    reproduces the token selection logic implemented in env/last-vit/conf.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_kernel = None

    @staticmethod
    def gaussian_kernel_1d(kernel_size: int, sigma: float) -> torch.Tensor:
        """Create a 1D Gaussian kernel normalized to max value 1."""
        coords = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel = kernel / kernel.max()
        return kernel

    def forward_to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Run the encoder and return only the patch tokens (exclude CLS)."""
        x = self._process_input(x)
        n = x.shape[0]
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        patches = x[:, 1:, :]
        return patches

    def smooth_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply the frequency-domain Gaussian smoothing used in conf.py."""
        if self.cached_kernel is None or self.cached_kernel.device != tokens.device:
            kernel_1d = self.gaussian_kernel_1d(tokens.shape[-1], tokens.shape[-1] ** 0.5)
            self.cached_kernel = kernel_1d.view(1, 1, -1).to(tokens.device)

        freq_tokens = torch.fft.fft(tokens, dim=-1)
        freq_tokens = torch.fft.fftshift(freq_tokens, dim=-1)
        freq_tokens = freq_tokens * self.cached_kernel
        freq_tokens = torch.fft.ifftshift(freq_tokens, dim=-1)
        smoothed = torch.fft.ifft(freq_tokens, dim=-1).real
        return smoothed


def load_model(device: torch.device) -> DenseViT:
    """Instantiate the dense ViT and load ImageNet pre-trained weights."""
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = DenseViT(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        dropout=0.0,
        attention_dropout=0.0,
        num_classes=1000,
        representation_size=None,
    )
    model.load_state_dict(weights.get_state_dict())
    model.eval().to(device)
    return model


def build_preprocess() -> Compose:
    """Return the preprocessing pipeline aligned with ViT-B/16."""
    return Compose(
        [
            Resize((224, 224), interpolation=Image.BICUBIC),
            ToTensor(),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def compute_channel_token_counts(
    model: DenseViT,
    image_tensor: torch.Tensor,
    per_channel_topk: int,
) -> torch.Tensor:
    """Aggregate how many channels select each token when applying top-k."""
    with torch.no_grad():
        patches = model.forward_to_patches(image_tensor)
        smoothed = model.smooth_tokens(patches)

    diff = patches / (smoothed - patches).abs().clamp_min(1e-6)
    num_tokens = diff.shape[1]
    k = min(per_channel_topk, num_tokens)

    # diff: [B, num_tokens, hidden_dim]
    _, indices = torch.topk(diff, k=k, dim=1, largest=True)
    # indices: [B, k, hidden_dim]
    flat_indices = indices.view(-1)
    counts = torch.bincount(flat_indices.cpu(), minlength=num_tokens).float()
    return counts


def build_token_masks(
    counts: torch.Tensor,
    num_tokens_to_select: int,
) -> torch.Tensor:
    """Return a binary mask of tokens selected by descending channel counts."""
    num_tokens = counts.shape[0]
    k = min(num_tokens_to_select, num_tokens)
    if k == 0 or counts.sum() == 0:
        return torch.zeros_like(counts)

    top_indices = torch.topk(counts, k=k, largest=True).indices
    mask = torch.zeros_like(counts)
    mask[top_indices] = 1.0
    return mask


def upscale_patch_map(patch_map: torch.Tensor, target_hw: tuple[int, int]) -> np.ndarray:
    """Upsample a [num_patches] vector to image resolution."""
    grid_size = int(math.sqrt(patch_map.numel()))
    patch_map_2d = patch_map.view(1, 1, grid_size, grid_size)
    upsampled = F.interpolate(patch_map_2d, size=target_hw, mode="nearest")
    return upsampled.squeeze().cpu().numpy()


def visualize_selection(
    image: Image.Image,
    mask_up: np.ndarray,
    counts_up: np.ndarray,
    selected_indices: list[int],
    output_file: Path,
    title: str,
    alpha: float = 0.45,
):
    """Save a visualization overlay highlighting selected tokens."""
    image_np = np.asarray(image).astype(np.float32) / 255.0
    if mask_up.max() > 0:
        counts_norm = counts_up / (counts_up.max() + 1e-6)
    else:
        counts_norm = counts_up * 0.0

    cmap = plt.get_cmap("inferno")
    heatmap = cmap(counts_norm)[..., :3] * mask_up[..., None]
    overlay = np.clip(image_np * (1 - alpha * mask_up[..., None]) + heatmap * alpha, 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_np)
    axes[0].set_title("Input")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    grid_size = int(math.sqrt(mask_up.size))
    patch_size = image.width // grid_size

    for token_idx in selected_indices:
        row = token_idx // grid_size
        col = token_idx % grid_size
        rect = plt.Rectangle(
            (col * patch_size, row * patch_size),
            patch_size,
            patch_size,
            linewidth=1.5,
            edgecolor="lime",
            facecolor="none",
        )
        axes[1].add_patch(rect)

    axes[1].set_title(title)
    axes[1].axis("off")
    fig.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize ViT token selection for different K values (token counts).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to an RGB image. It will be resized to 224x224.",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        required=True,
        help="List of token counts (K) to visualize.",
    )
    parser.add_argument(
        "--per-channel-topk",
        type=int,
        default=1,
        help="Number of top tokens selected per channel before aggregating counts.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("videollm-online/visualize/output"),
        help="Directory to store the visualizations.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = load_model(device)
    preprocess = build_preprocess()

    image = Image.open(args.image).convert("RGB")
    image_resized = image.resize((224, 224), resample=Image.BICUBIC)
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    counts = compute_channel_token_counts(
        model=model,
        image_tensor=image_tensor,
        per_channel_topk=max(args.per_channel_topk, 1),
    )

    grid_size = int(math.sqrt(counts.numel()))
    if grid_size * grid_size != counts.numel():
        raise ValueError(f"Token count {counts.numel()} is not a perfect square.")

    counts_up = upscale_patch_map(counts, image_resized.size[::-1])

    for k in args.k_values:
        mask = build_token_masks(counts, num_tokens_to_select=max(k, 0))
        mask_up = upscale_patch_map(mask, image_resized.size[::-1])

        selected_indices = torch.nonzero(mask, as_tuple=False).view(-1).tolist()
        title = f"K={k} tokens (per-channel top-k={args.per_channel_topk})"
        output_path = args.output_dir / f"token_selection_K{k}_pctop{args.per_channel_topk}.png"

        visualize_selection(
            image=image_resized,
            mask_up=mask_up,
            counts_up=counts_up,
            selected_indices=selected_indices,
            output_file=output_path,
            title=title,
        )

        print(
            f"[INFO] Saved visualization for K={k} to {output_path}. "
            f"Selected tokens: {selected_indices}"
        )


if __name__ == "__main__":
    main()

