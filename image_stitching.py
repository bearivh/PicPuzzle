import sys
import cv2
import numpy as np


# ─────────────────────────────────────────────
# 1. Feature detection & matching
# ─────────────────────────────────────────────

def detect_and_describe(img, max_features=5000):
    """Detect SIFT keypoints and compute descriptors."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_features)
    kps, des = sift.detectAndCompute(gray, None)
    return kps, des


def match_features(des1, des2, ratio=0.75):
    """Match descriptors using BFMatcher + Lowe's ratio test."""
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]
    return good


def find_homography(kps1, kps2, matches, min_matches=10):
    """
    Estimate homography H such that:  pts2 ≈ H @ pts1
    Returns H, mask  (None, None if not enough matches).
    """
    if len(matches) < min_matches:
        return None, None
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    return H, mask


# ─────────────────────────────────────────────
# 2. Auto-ordering: build a chain of images
# ─────────────────────────────────────────────

def build_chain(images):
    """
    Given N images, find a linear ordering by counting inlier matches
    between every pair and greedily chaining from the best anchor.

    Returns ordered list of (original_index, image).
    """
    n = len(images)
    scale = 0.25
    kps_list, des_list = [], []
    for img in images:
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(w * scale), int(h * scale)))
        kps, des = detect_and_describe(small)
        kps_list.append(kps)
        des_list.append(des)

    score = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if des_list[i] is None or des_list[j] is None:
                continue
            good = match_features(des_list[i], des_list[j])
            score[i, j] = score[j, i] = len(good)

    print("\n[Match scores between image pairs]")
    for i in range(n):
        for j in range(i + 1, n):
            if score[i, j] > 0:
                print(f"  image{i+1} <-> image{j+1}: {score[i,j]} matches")

    best = np.unravel_index(score.argmax(), score.shape)
    chain = list(best)          # e.g. [0, 3]
    used = set(chain)

    def best_neighbour(anchor, exclude):
        candidates = [(score[anchor, k], k) for k in range(n) if k not in exclude]
        if not candidates:
            return None
        return max(candidates)[1]

    while len(chain) < n:
        right = best_neighbour(chain[-1], used)
        left  = best_neighbour(chain[0],  used)

        r_score = score[chain[-1], right] if right is not None else 0
        l_score = score[chain[0],  left]  if left  is not None else 0

        if r_score >= l_score and right is not None:
            chain.append(right)
            used.add(right)
        elif left is not None:
            chain.insert(0, left)
            used.add(left)
        else:
            remaining = [k for k in range(n) if k not in used]
            chain.append(remaining[0])
            used.add(remaining[0])

    print(f"\n[Auto-detected stitching order]: {[f'image{i+1}' for i in chain]}")
    return [(i, images[i]) for i in chain]


# ─────────────────────────────────────────────
# 3. Multi-band blending (Laplacian pyramid)
# ─────────────────────────────────────────────

def gaussian_pyramid(img, levels):
    gp = [img.astype(np.float32)]
    for _ in range(levels):
        gp.append(cv2.pyrDown(gp[-1]))
    return gp


def laplacian_pyramid(img, levels):
    gp = gaussian_pyramid(img, levels)
    lp = []
    for i in range(levels):
        up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        lp.append(gp[i] - up)
    lp.append(gp[levels])
    return lp


def blend_pyramids(lp1, lp2, mask_pyr):
    blended = []
    for l1, l2, m in zip(lp1, lp2, mask_pyr):
        blended.append(l1 * m + l2 * (1 - m))
    return blended


def reconstruct(lp):
    img = lp[-1]
    for level in reversed(lp[:-1]):
        img = cv2.pyrUp(img, dstsize=(level.shape[1], level.shape[0]))
        img = img + level
    return img


def multiband_blend(base, warped, mask_base, mask_warped, levels=4):
    overlap = (mask_base > 0) & (mask_warped > 0)
    seam_mask = np.where(overlap, 0.5, mask_base.astype(np.float32))
    seam_mask = cv2.GaussianBlur(seam_mask, (0, 0), sigmaX=30)
    seam_mask = np.clip(seam_mask, 0, 1)
    seam_3 = np.dstack([seam_mask] * 3)

    base_f   = base.astype(np.float32)
    warped_f = warped.astype(np.float32)

    lp1 = laplacian_pyramid(base_f,   levels)
    lp2 = laplacian_pyramid(warped_f, levels)

    mask_gp = gaussian_pyramid(seam_3, levels)
    mask_pyr = mask_gp[:levels] + [mask_gp[levels]]

    blended_lp = blend_pyramids(lp1, lp2, mask_pyr)
    result = reconstruct(blended_lp)
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


# ─────────────────────────────────────────────
# 4. Pairwise stitching
# ─────────────────────────────────────────────

def stitch_pair(img_base, img_src):
    kps1, des1 = detect_and_describe(img_base)
    kps2, des2 = detect_and_describe(img_src)

    matches = match_features(des1, des2)
    print(f"  -> {len(matches)} good matches")

    H, mask = find_homography(kps1, kps2, matches)
    if H is None:
        print("  [WARNING] Not enough matches to stitch this pair – skipping.")
        return img_base

    h1, w1 = img_base.shape[:2]
    h2, w2 = img_src.shape[:2]

    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)

    H_inv = np.linalg.inv(H)
    warped_corners = cv2.perspectiveTransform(corners2, H_inv)

    all_corners = np.concatenate([corners1, warped_corners], axis=0)
    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    tx, ty = -x_min, -y_min
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min

    MAX_DIM = 15000
    if canvas_w > MAX_DIM or canvas_h > MAX_DIM:
        scale = MAX_DIM / max(canvas_w, canvas_h)
        canvas_w = int(canvas_w * scale)
        canvas_h = int(canvas_h * scale)
        tx = int(tx * scale)
        ty = int(ty * scale)
        S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
        H_inv = S @ H_inv
        h1 = int(h1 * scale); w1 = int(w1 * scale)
        img_base = cv2.resize(img_base, (w1, h1))
        h2 = int(h2 * scale); w2 = int(w2 * scale)
        img_src  = cv2.resize(img_src,  (w2, h2))

    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)

    warped_src = cv2.warpPerspective(img_src, T @ H_inv, (canvas_w, canvas_h))

    canvas_base = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_base[ty:ty + h1, tx:tx + w1] = img_base

    mask_base   = (canvas_base.sum(axis=2) > 0).astype(np.float32)
    mask_warped = (warped_src.sum(axis=2)  > 0).astype(np.float32)

    result = multiband_blend(canvas_base, warped_src, mask_base, mask_warped, levels=4)

    only_base   = (mask_base > 0) & (mask_warped == 0)
    only_warped = (mask_warped > 0) & (mask_base == 0)
    result[only_base]   = canvas_base[only_base]
    result[only_warped] = warped_src[only_warped]

    return result


# ─────────────────────────────────────────────
# 5. Crop black borders
# ─────────────────────────────────────────────

def crop_black_borders(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y + h, x:x + w]


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────

def main(image_paths):
    print(f"Loading {len(image_paths)} images ...")
    images = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[ERROR] Cannot load: {p}")
            sys.exit(1)
        h, w = img.shape[:2]
        max_dim = 1600
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        images.append(img)
    print("Done loading.\n")

    ordered = build_chain(images)

    print("\n[Stitching images in order ...]")
    panorama = ordered[0][1]
    for idx, (orig_idx, img) in enumerate(ordered[1:], start=1):
        print(f"\nStitching image{orig_idx+1} ({idx}/{len(ordered)-1}) ...")
        panorama = stitch_pair(panorama, img)

    panorama = crop_black_borders(panorama)

    output_path = "result.jpg"
    cv2.imwrite(output_path, panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\n[Done] Panorama saved to: {output_path}")
    print(f"       Size: {panorama.shape[1]} x {panorama.shape[0]} px")
    return panorama


if __name__ == "__main__":
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    else:
        import glob
        paths = sorted(glob.glob("image*.jpg"))[:7]
        if not paths:
            print("Usage: python image_stitching.py img1.jpg img2.jpg ...")
            sys.exit(1)

    main(paths)