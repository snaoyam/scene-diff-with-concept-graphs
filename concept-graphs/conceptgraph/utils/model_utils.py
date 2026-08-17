import numpy as np
import scipy.ndimage as ndi
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

# Official DINOv3 image transform for LVD-1689M weights (dinov3/README.md "Image
# transforms" section) -- resize_size=256 is the README's own default.
dinov3_preprocess = v2.Compose([
    v2.ToImage(),
    v2.Resize((256, 256), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

# @profile
def compute_clip_features_batched(image, detections, clip_model, clip_preprocess, clip_tokenizer, classes, device):

    image = Image.fromarray(image)
    padding = 20  # Adjust the padding amount as needed

    image_crops = []
    preprocessed_images = []
    text_tokens = []

    # Prepare data for batch processing
    for idx in range(len(detections.xyxy)):
        x_min, y_min, x_max, y_max = detections.xyxy[idx]
        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        preprocessed_image = clip_preprocess(cropped_image).unsqueeze(0)
        preprocessed_images.append(preprocessed_image)

        class_id = detections.class_id[idx]
        text_tokens.append(classes[class_id])
        image_crops.append(cropped_image)

    if not preprocessed_images:
        # No detections in this frame -- torch.cat() below has nothing to concatenate.
        # image_feats is never indexed by callers when there are 0 detections for the
        # frame (they loop over range(len(gobs['mask']))), so shape doesn't matter here.
        return image_crops, np.empty((0, 0), dtype=np.float32), []

    # Convert lists to batches
    # dtype=torch.float16 matches clip_model's precision="fp16" (open_clip's manual
    # mixed-precision mode) -- encode_image() doesn't cast its input itself.
    preprocessed_images_batch = torch.cat(preprocessed_images, dim=0).to(device, dtype=torch.float16)
    text_tokens_batch = clip_tokenizer(text_tokens).to(device)

    # Batch inference
    with torch.no_grad():
        image_features = clip_model.encode_image(preprocessed_images_batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # text_features = clip_model.encode_text(text_tokens_batch)
        # text_features /= text_features.norm(dim=-1, keepdim=True)

    # Convert to numpy
    image_feats = image_features.cpu().numpy()
    # text_feats = text_features.cpu().numpy()
    # image_feats = []
    text_feats = []

    return image_crops, image_feats, text_feats


def compute_dinov3_dense_features(image_rgb, dinov3_model, device):
    '''
    Single DINOv3 forward pass on the whole frame; patch tokens reshaped to a
    spatial grid and bilinearly upsampled to the frame's original (H, W) so
    any downstream mask can pool a per-object embedding directly (see
    pool_dinov3_features_by_mask) -- avoids the CLS-token-of-a-padded-crop
    contamination problem, since every pixel's feature comes from its own
    exact location in the full frame rather than a crop that includes
    whatever happens to surround a small object.

    Returns (D, H, W) L2-normalized torch tensor on `device`.
    '''
    h, w = image_rgb.shape[:2]
    x = dinov3_preprocess(image_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        patch_tokens = dinov3_model.forward_features(x)["x_norm_patchtokens"]
    n_side = int(patch_tokens.shape[1] ** 0.5)
    feat_grid = patch_tokens.reshape(1, n_side, n_side, -1).permute(0, 3, 1, 2)
    dense = F.interpolate(feat_grid, size=(h, w), mode="bilinear", align_corners=False)[0]
    return dense / dense.norm(dim=0, keepdim=True)


def pool_dinov3_features_by_mask(dense_features, masks, erosion_iterations=5):
    '''
    Per-mask average of dense_features (see compute_dinov3_dense_features),
    eroding each mask first to drop boundary pixels -- mask edges sit right
    against a neighboring object or background, so their pooled-in feature
    (upsampled from a coarse patch grid) isn't representative of the
    object's own appearance. Falls back to the un-eroded mask if erosion
    empties it (thin/small objects).

    masks: (N, H, W) boolean. Returns (N, D) L2-normalized numpy array.
    '''
    if len(masks) == 0:
        return np.empty((0, 0), dtype=np.float32)

    feats = []
    for mask in masks:
        eroded = ndi.binary_erosion(mask, iterations=erosion_iterations)
        region = eroded if eroded.any() else mask
        region_t = torch.from_numpy(region).to(dense_features.device)
        feat = dense_features[:, region_t].mean(dim=1)
        feats.append((feat / feat.norm()).cpu().numpy())
    return np.stack(feats)
