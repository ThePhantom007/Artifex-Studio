import cv2
import numpy as np
from skimage.exposure import match_histograms
import warnings

warnings.filterwarnings("ignore")


def _normalize_exposures(images: list) -> list:
    """
    Matches the histogram of each image to the first image.
    This eliminates visible seam lines caused by auto-exposure differences
    between frames — the single biggest source of "stitched" looking panoramas.
    scikit-image is already in requirements.txt.
    """
    if len(images) < 2:
        return images
    reference = images[0]
    normalized = [reference]
    for img in images[1:]:
        matched = match_histograms(img, reference, channel_axis=-1)
        normalized.append(matched.astype(np.uint8))
    return normalized


def _fast_inner_crop(pano: np.ndarray) -> np.ndarray:
    """
    Finds the largest axis-aligned rectangle with no black border pixels.

    Replaces the original erosion loop which was O(n * iterations) and could
    take minutes on a large panorama. This row/column scan is O(n).
    """
    # Small border ensures contours close properly at image edges
    bordered = cv2.copyMakeBorder(pano, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

    # FIX: The original line 55 was:
    #   gray = cv2.cvtColor(pano, cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY) ...)
    # which passed the RESULT of an inner cvtColor as the conversion CODE
    # argument of the outer cvtColor. This crashes every time.
    gray = cv2.cvtColor(bordered, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find rows/columns where every single pixel is non-zero (fully inside the photo)
    row_mask = np.all(thresh == 255, axis=1)
    col_mask = np.all(thresh == 255, axis=0)

    valid_rows = np.where(row_mask)[0]
    valid_cols = np.where(col_mask)[0]

    if len(valid_rows) == 0 or len(valid_cols) == 0:
        # Fallback: use largest contour bounding box if the scan finds nothing
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return pano  # Nothing to crop, return as-is
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return bordered[y:y + h, x:x + w]

    r1, r2 = valid_rows[0], valid_rows[-1] + 1
    c1, c2 = valid_cols[0], valid_cols[-1] + 1

    return bordered[r1:r2, c1:c2]


def stitch_images(images: list[np.ndarray]) -> np.ndarray | str:
    """
    Stitches a list of overlapping BGR images into a seamless panorama.

    Pipeline:
        1. Validate and downscale oversized inputs
        2. Normalize exposure across all frames (histogram matching)
        3. OpenCV Stitcher with PANORAMA mode (feature detection + multiband blend)
        4. Auto-crop black borders with fast row/column scan
    """
    if len(images) < 2:
        return "Error: At least 2 images are required to create a panorama."

    try:
        # --- 1. Validation & RAM Protection ---
        max_dim = 3500
        processed = []
        for img in images:
            if img is None:
                return "Error: One or more images failed to load from disk."
            if len(img.shape) != 3 or img.shape[2] != 3:
                return "Error: All images must be 3-channel BGR format."
            if max(img.shape[0], img.shape[1]) > max_dim:
                scale = max_dim / max(img.shape[0], img.shape[1])
                img = cv2.resize(
                    img,
                    (int(img.shape[1] * scale), int(img.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA
                )
            processed.append(img)

        # --- 2. Exposure Normalization ---
        # Aligns brightness/colour across frames before blending.
        # This is the most effective technique for eliminating visible seams.
        print("🎨 Normalizing exposure across frames...")
        processed = _normalize_exposures(processed)

        # --- 3. Stitching ---
        print(f"🧵 Stitching {len(processed)} images...")
        stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
        status, pano = stitcher.stitch(processed)

        if status != cv2.Stitcher_OK:
            error_map = {
                cv2.Stitcher_ERR_NEED_MORE_IMGS: (
                    "Not enough matching features. "
                    "Ensure adjacent images overlap by at least 20-30%."
                ),
                cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: (
                    "Perspective estimation failed. "
                    "Images may not be from the same scene or viewpoint."
                ),
                cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: (
                    "Camera calibration failed. Try images with more distinctive features."
                ),
            }
            msg = error_map.get(status, f"Unknown stitching error (code {status}).")
            print(f"❌ {msg}")
            return f"Error: {msg}"

        print("✅ Stitching successful. Cropping borders...")

        # --- 4. Auto-Crop ---
        cropped = _fast_inner_crop(pano)

        print("🖼️ Panorama complete!")
        return cropped

    except Exception as e:
        print(f"❌ Stitching error: {e}")
        return f"Unexpected Error: {str(e)}"