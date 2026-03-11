import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def _try_opencv_stitcher(images_bgr: list, mode) -> np.ndarray | None:
    """Attempt OpenCV Stitcher in the given mode. Returns BGR array or None."""
    stitcher = cv2.Stitcher.create(mode)
    stitcher.setPanoConfidenceThresh(0.3)   # lower = more permissive matching
    status, result = stitcher.stitch(images_bgr)
    if status == cv2.Stitcher_OK and result is not None:
        return result
    codes = {
        cv2.Stitcher_ERR_NEED_MORE_IMGS:    "need more images",
        cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "homography failed",
        cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "camera params failed",
    }
    logger.warning(f"Stitcher failed: {codes.get(status, f'code {status}')}")
    return None


def _manual_stitch(images_bgr: list) -> np.ndarray | None:
    """
    Manual SIFT + FLANN + homography stitching pipeline.
    More robust than OpenCV Stitcher for indoor scenes with repetitive
    patterns, because we control RANSAC thresholds and can stitch
    incrementally (image by image) rather than all-at-once.
    """
    sift = cv2.SIFT_create(
        nfeatures=5000,       # more features → better chance on textured walls
        contrastThreshold=0.02,
        edgeThreshold=15,
    )

    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        {"algorithm": FLANN_INDEX_KDTREE, "trees": 5},
        {"checks": 100},
    )

    # Start with the first image as the base
    base = images_bgr[0]

    for i, next_img in enumerate(images_bgr[1:], start=1):
        kp1, des1 = sift.detectAndCompute(base,     None)
        kp2, des2 = sift.detectAndCompute(next_img, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            logger.error(f"Not enough keypoints for pair 0→{i}")
            return None

        # Lowe's ratio test — filters ambiguous matches
        matches = flann.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.72 * n.distance]
        logger.info(f"Pair 0→{i}: {len(good)} good matches from {len(matches)}")

        if len(good) < 10:
            logger.error(f"Insufficient good matches ({len(good)}) for pair 0→{i}. "
                         f"Ensure images overlap by 25–35%.")
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # RANSAC homography — higher reprojection threshold for indoor scenes
        H, mask = cv2.findHomography(
            dst_pts, src_pts,
            cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=3000,
            confidence=0.995,
        )

        if H is None or mask is None or mask.sum() < 8:
            logger.error(f"Homography estimation failed for pair 0→{i} "
                         f"(inliers: {mask.sum() if mask is not None else 0})")
            return None

        # Warp next_img into base coordinate space
        h1, w1 = base.shape[:2]
        h2, w2 = next_img.shape[:2]

        corners = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
        warped_corners = cv2.perspectiveTransform(corners, H)
        all_corners = np.concatenate([
            np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2),
            warped_corners,
        ])

        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())

        # Translation matrix to keep all content in positive coordinates
        T = np.array([[1, 0, -x_min],
                      [0, 1, -y_min],
                      [0, 0, 1]], dtype=np.float64)

        out_w = x_max - x_min
        out_h = y_max - y_min

        # Guard against unreasonable canvas sizes (bad homography)
        if out_w > 20000 or out_h > 20000 or out_w <= 0 or out_h <= 0:
            logger.error(f"Unreasonable output canvas {out_w}×{out_h} — homography is degenerate")
            return None

        warped = cv2.warpPerspective(next_img, T @ H, (out_w, out_h))

        # Place base onto the canvas
        canvas = warped.copy()
        canvas[-y_min:h1 - y_min, -x_min:w1 - x_min] = base

        # Simple feather blend in the overlap region
        base_mask   = np.zeros((out_h, out_w), np.float32)
        warped_mask = np.zeros((out_h, out_w), np.float32)

        base_mask[-y_min:h1-y_min, -x_min:w1-x_min] = 1.0
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped_mask[warped_gray > 0] = 1.0

        overlap = (base_mask > 0) & (warped_mask > 0)
        if overlap.any():
            # In overlap region, blend 60% base / 40% warped for natural seam
            alpha = 0.6
            canvas[overlap] = (
                alpha       * base_mask[overlap, None]   * canvas[overlap].astype(np.float32) +
                (1 - alpha) * warped_mask[overlap, None] * warped[overlap].astype(np.float32)
            ).clip(0, 255).astype(np.uint8)

        base = canvas
        logger.info(f"After stitching image {i}: canvas is {base.shape[1]}×{base.shape[0]}px")

    return base


def _histogram_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match source histogram to reference per channel."""
    matched = np.zeros_like(source)
    for c in range(3):
        src_hist, _ = np.histogram(source[:,:,c].flatten(), 256, [0,256])
        ref_hist, _ = np.histogram(reference[:,:,c].flatten(), 256, [0,256])
        src_cdf = src_hist.cumsum() / src_hist.sum()
        ref_cdf = ref_hist.cumsum() / ref_hist.sum()
        lut = np.interp(src_cdf, ref_cdf, np.arange(256))
        matched[:,:,c] = lut[source[:,:,c]]
    return matched.astype(np.uint8)


def _autocrop(img_bgr: np.ndarray) -> np.ndarray:
    """Crop black borders left by warping."""
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return img_bgr[y:y+h, x:x+w]


def stitch_images(image_arrays: list) -> np.ndarray | str:
    if len(image_arrays) < 2:
        return "Error: At least 2 images are required."
    if len(image_arrays) > 20:
        return "Error: Maximum 20 images supported."

    # Convert RGB → BGR for OpenCV
    images_bgr = [img[:, :, ::-1].copy() for img in image_arrays]

    # Histogram-match all images to the first for consistent exposure
    reference = images_bgr[0]
    images_bgr = [reference] + [
        _histogram_match(img, reference) for img in images_bgr[1:]
    ]

    logger.info(f"Stitching {len(images_bgr)} images...")

    result_bgr = None

    # ── Attempt 1: OpenCV Stitcher SCANS mode (best for indoor/planar) ──
    logger.info("Trying OpenCV Stitcher — SCANS mode...")
    result_bgr = _try_opencv_stitcher(images_bgr, cv2.Stitcher_SCANS)

    # ── Attempt 2: OpenCV Stitcher PANORAMA mode ─────────────────────
    if result_bgr is None:
        logger.info("Trying OpenCV Stitcher — PANORAMA mode...")
        result_bgr = _try_opencv_stitcher(images_bgr, cv2.Stitcher_PANORAMA)

    # ── Attempt 3: Manual SIFT + homography pipeline ─────────────────
    if result_bgr is None:
        logger.info("Trying manual SIFT + homography pipeline...")
        result_bgr = _manual_stitch(images_bgr)

    if result_bgr is None:
        return (
            "Error: All stitching methods failed. Tips:\n"
            "• Ensure adjacent frames overlap by 25–35%\n"
            "• Avoid purely repetitive surfaces as the main feature (e.g. plain walls)\n"
            "• Keep the camera level between shots\n"
            "• Include distinctive objects (furniture, clock, curtain) in the overlap region"
        )

    # Autocrop black borders, convert back to RGB
    result_bgr  = _autocrop(result_bgr)
    result_rgb  = result_bgr[:, :, ::-1].copy()

    logger.info(f"Stitch complete: {result_rgb.shape[1]}×{result_rgb.shape[0]}px")
    return result_rgb.astype(np.uint8)