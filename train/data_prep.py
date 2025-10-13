import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

# =========================
# CONFIG
# =========================
DATA_DIR = Path("../data")
OUTPUT_DIR = Path("../data_prepared")

# Use "hed" or "canny"
EDGE_METHOD = "hed"

# Optional: Resize all images for consistency (Pix2Pix expects same size)
TARGET_SIZE = (256, 256)

# =========================
# HED Edge Detector Setup
# =========================
def load_hed_model():
    proto_url = "https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/hed_deploy.prototxt"
    weights_url = "https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/hed_pretrained_bsds.caffemodel"

    proto_path = "hed_deploy.prototxt"
    weights_path = "hed_pretrained_bsds.caffemodel"

    if not os.path.exists(proto_path):
        import urllib.request
        print("Downloading HED model...")
        urllib.request.urlretrieve(proto_url, proto_path)
        urllib.request.urlretrieve(weights_url, weights_path)
    return cv2.dnn.readNetFromCaffe(proto_path, weights_path)

hed_net = load_hed_model() if EDGE_METHOD == "hed" else None


# =========================
# EDGE FUNCTIONS
# =========================
def hed_edge(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(w, h),
                                 mean=(104.00698793, 116.66876762, 122.67891434),
                                 swapRB=False, crop=False)
    hed_net.setInput(blob)
    edge = hed_net.forward()[0, 0]
    edge = cv2.resize(edge, (w, h))
    edge = (255 * edge).astype(np.uint8)
    return edge


def canny_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 100, 200)
    return edge


def edge_detect(image):
    return hed_edge(image) if EDGE_METHOD == "hed" else canny_edge(image)


# =========================
# PROCESS FUNCTION
# =========================
def process_split(split):
    input_dir = DATA_DIR / split
    output_input = OUTPUT_DIR / split / "inputs"
    output_target = OUTPUT_DIR / split / "targets"
    output_input.mkdir(parents=True, exist_ok=True)
    output_target.mkdir(parents=True, exist_ok=True)

    for species_dir in sorted(input_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        for img_path in tqdm(list(species_dir.glob("*.*")), desc=f"{split}/{species_dir.name}"):
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                if TARGET_SIZE:
                    image = cv2.resize(image, TARGET_SIZE)

                edge = edge_detect(image)

                out_name = f"{species_dir.name}_{img_path.stem}.jpg"
                cv2.imwrite(str(output_input / out_name), edge)
                cv2.imwrite(str(output_target / out_name), image)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    for split in ["train", "test", "valid"]:
        if (DATA_DIR / split).exists():
            process_split(split)
