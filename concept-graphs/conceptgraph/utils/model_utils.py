import numpy as np
import torch
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


def compute_dinov3_features_batched(image_crops, dinov3_model, device):
    '''
    DINOv3 appearance embedding per detection, reusing the same padded crops
    compute_clip_features_batched() already produced (image_crops) so both
    embeddings come from the identical crop region. Returns the global
    embedding (x_norm_clstoken), L2-normalized like clip_ft.
    '''
    if not image_crops:
        return np.empty((0, 0), dtype=np.float32)

    preprocessed_images_batch = torch.cat(
        [dinov3_preprocess(crop).unsqueeze(0) for crop in image_crops], dim=0
    ).to(device, dtype=torch.float16)

    with torch.no_grad():
        dino_features = dinov3_model.forward_features(preprocessed_images_batch)
        cls_token = dino_features["x_norm_clstoken"]
        cls_token /= cls_token.norm(dim=-1, keepdim=True)

    return cls_token.cpu().numpy()
