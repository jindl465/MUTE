import os

import cv2
import numpy as np


FACE_LANDMARKER_PATH = (
    "./pretrained_models/face_analysis/models/"
    "face_landmarker_v2_with_blendshapes.task"
)

silhouette_ids = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

lip_ids = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91,
    181, 84, 17, 314, 405, 321, 375,
]


def _load_mediapipe():
    import mediapipe as mp

    return mp


def compute_face_landmarks(detection_result, height, width):
    face_landmarks_list = detection_result.face_landmarks
    if len(face_landmarks_list) != 1:
        print("#face is invalid:", len(face_landmarks_list))
        return []
    return [[p.x * width, p.y * height] for p in face_landmarks_list[0]]


def get_landmark(file):
    mp = _load_mediapipe()
    base_options = mp.tasks.BaseOptions
    face_landmarker = mp.tasks.vision.FaceLandmarker
    face_landmarker_options = mp.tasks.vision.FaceLandmarkerOptions
    vision_running_mode = mp.tasks.vision.RunningMode
    options = face_landmarker_options(
        base_options=base_options(model_asset_path=FACE_LANDMARKER_PATH),
        running_mode=vision_running_mode.IMAGE,
    )

    with face_landmarker.create_from_options(options) as landmarker:
        image = mp.Image.create_from_file(str(file))
        height, width = image.height, image.width
        result = landmarker.detect(image)
        face_landmark = compute_face_landmarks(result, height, width)

    return np.array(face_landmark), height, width


def get_landmark_overframes(landmark_model, frames_path):
    mp = _load_mediapipe()
    face_landmarks = []
    height = width = None
    for file in sorted(os.listdir(frames_path)):
        image = mp.Image.create_from_file(os.path.join(frames_path, file))
        height, width = image.height, image.width
        result = landmark_model.detect(image)
        face_landmarks.append(compute_face_landmarks(result, height, width))
    return face_landmarks, height, width


def expand_region(region, image_w, image_h, expand_ratio=1.0):
    min_x, max_x, min_y, max_y = region
    mid_x = (max_x + min_x) // 2
    side_len_x = (max_x - min_x) * expand_ratio
    mid_y = (max_y + min_y) // 2
    side_len_y = (max_y - min_y) * expand_ratio
    min_x = mid_x - side_len_x // 2
    max_x = mid_x + side_len_x // 2
    min_y = mid_y - side_len_y // 2
    max_y = mid_y + side_len_y // 2

    if min_x < 0:
        max_x -= min_x
        min_x = 0
    if max_x > image_w:
        min_x -= max_x - image_w
        max_x = image_w
    if min_y < 0:
        max_y -= min_y
        min_y = 0
    if max_y > image_h:
        min_y -= max_y - image_h
        max_y = image_h

    return round(min_x), round(max_x), round(min_y), round(max_y)


def get_lip_mask(landmarks, height, width, out_path=None, expand_ratio=2.0):
    lip_landmarks = np.take(landmarks, lip_ids, 0)
    min_xy_lip = np.round(np.min(lip_landmarks, 0))
    max_xy_lip = np.round(np.max(lip_landmarks, 0))
    min_xy_lip[0], max_xy_lip[0], min_xy_lip[1], max_xy_lip[1] = expand_region(
        [min_xy_lip[0], max_xy_lip[0], min_xy_lip[1], max_xy_lip[1]],
        width,
        height,
        expand_ratio,
    )
    lip_mask = np.zeros((height, width), dtype=np.uint8)
    lip_mask[
        round(min_xy_lip[1]):round(max_xy_lip[1]),
        round(min_xy_lip[0]):round(max_xy_lip[0]),
    ] = 255
    if out_path:
        cv2.imwrite(str(out_path), lip_mask)
        return None
    return lip_mask


def get_face_mask(landmarks, height, width, out_path=None, expand_ratio=1.2):
    face_landmarks = np.take(landmarks, silhouette_ids, 0)
    min_xy_face = np.round(np.min(face_landmarks, 0))
    max_xy_face = np.round(np.max(face_landmarks, 0))
    min_xy_face[0], max_xy_face[0], min_xy_face[1], max_xy_face[1] = (
        expand_region(
            [min_xy_face[0], max_xy_face[0], min_xy_face[1], max_xy_face[1]],
            width,
            height,
            expand_ratio,
        )
    )
    face_mask = np.zeros((height, width), dtype=np.uint8)
    face_mask[
        round(min_xy_face[1]):round(max_xy_face[1]),
        round(min_xy_face[0]):round(max_xy_face[0]),
    ] = 255
    if out_path:
        cv2.imwrite(str(out_path), face_mask)
        return None
    return face_mask


def get_union_mask(masks):
    union_mask = None
    for mask in masks:
        union_mask = mask if union_mask is None else np.maximum(union_mask, mask)

    if union_mask is not None:
        rows = np.any(union_mask, axis=1)
        cols = np.any(union_mask, axis=0)
        try:
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
        except Exception as exc:
            print(str(exc))
            return 0.0
        union_mask[ymin:ymax + 1, xmin:xmax + 1] = np.max(union_mask)
    return union_mask


def get_union_lip_mask(landmarks, height, width, expand_ratio=1):
    lip_masks = [
        get_lip_mask(
            landmarks=landmark,
            height=height,
            width=width,
            expand_ratio=expand_ratio,
        )
        for landmark in landmarks
    ]
    return get_union_mask(lip_masks)


def get_union_face_mask(landmarks, height, width, expand_ratio=1):
    face_masks = [
        get_face_mask(
            landmarks=landmark,
            height=height,
            width=width,
            expand_ratio=expand_ratio,
        )
        for landmark in landmarks
    ]
    return get_union_mask(face_masks)


def blur_mask(mask, resize_dim=(64, 64), kernel_size=(51, 51)):
    if mask is None:
        return None
    resized_mask = cv2.resize(mask, resize_dim)
    blurred_mask = cv2.GaussianBlur(resized_mask, kernel_size, 0)
    return cv2.normalize(blurred_mask, None, 0, 255, cv2.NORM_MINMAX)


def get_blur_mask(
    file_path,
    output_file_path,
    resize_dim=(64, 64),
    kernel_size=(101, 101),
):
    mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return f"Failed to load image: {file_path}"
    normalized_mask = blur_mask(
        mask,
        resize_dim=resize_dim,
        kernel_size=kernel_size,
    )
    cv2.imwrite(output_file_path, normalized_mask)
    return f"Processed and saved: {output_file_path}"


def get_background_mask(file_path, output_file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Failed to load image: {file_path}")
        return
    inverted_image = 1.0 - (image / 255.0)
    inverted_image = (inverted_image * 255).astype(np.uint8)
    cv2.imwrite(output_file_path, inverted_image)


def get_sep_face_mask(file_path1, file_path2, output_file_path):
    mask1 = cv2.imread(file_path1, cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(file_path2, cv2.IMREAD_GRAYSCALE)
    if mask1 is None or mask2 is None:
        print(f"Failed to load images: {file_path1}, {file_path2}")
        return
    if mask1.shape != mask2.shape:
        print(
            f"Image shapes do not match for {file_path1}: "
            f"{mask1.shape} vs {mask2.shape}"
        )
        return
    cv2.imwrite(output_file_path, cv2.subtract(mask1, mask2))


def get_mask(file, cache_dir, face_expand_ratio):
    os.makedirs(cache_dir, exist_ok=True)
    file_name = os.path.basename(file).split(".")[0]
    lip_mask_path = os.path.join(cache_dir, f"{file_name}_lip_mask.png")
    face_mask_path = os.path.join(cache_dir, f"{file_name}_face_mask.png")
    face_blur_path = os.path.join(cache_dir, f"{file_name}_face_mask_blur.png")
    sep_lip_path = os.path.join(cache_dir, f"{file_name}_sep_lip.png")
    sep_background_path = os.path.join(cache_dir, f"{file_name}_sep_background.png")
    sep_face_path = os.path.join(cache_dir, f"{file_name}_sep_face.png")

    required_paths = [
        lip_mask_path,
        face_mask_path,
        sep_lip_path,
        sep_background_path,
        sep_face_path,
    ]
    if all(os.path.exists(path) for path in required_paths):
        return

    if os.path.exists(lip_mask_path) and os.path.exists(face_mask_path):
        get_blur_mask(face_mask_path, face_blur_path, kernel_size=(51, 51))
        get_blur_mask(lip_mask_path, sep_lip_path, kernel_size=(31, 31))
        get_background_mask(face_blur_path, sep_background_path)
        get_sep_face_mask(face_blur_path, sep_lip_path, sep_face_path)
        return

    landmarks, height, width = get_landmark(file)

    get_lip_mask(landmarks, height, width, lip_mask_path)
    get_face_mask(landmarks, height, width, face_mask_path, face_expand_ratio)
    get_blur_mask(face_mask_path, face_blur_path, kernel_size=(51, 51))
    get_blur_mask(lip_mask_path, sep_lip_path, kernel_size=(31, 31))
    get_background_mask(face_blur_path, sep_background_path)
    get_sep_face_mask(face_blur_path, sep_lip_path, sep_face_path)
