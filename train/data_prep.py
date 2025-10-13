import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

# =========================
# CONFIG
# =========================
DATA_DIR = Path("/workspace/data")
OUTPUT_DIR = Path("/workspace/data_prepared")

# Use "hed" or "canny"
EDGE_METHOD = "canny"

# Optional: Resize all images for consistency (Pix2Pix expects same size)
TARGET_SIZE = (256, 256)

# =========================
# HED Edge Detector Setup
# =========================
def load_hed_model():
    import urllib.request

    proto_url = "https://raw.githubusercontent.com/s9xie/hed/master/examples/hed/deploy.prototxt"
    weights_url = "https://pjreddie.com/media/files/hed_pretrained_bsds.caffemodel"

    proto_path = "/workspace/train/hed_deploy.prototxt"
    weights_path = "/workspace/train/hed_pretrained_bsds.caffemodel"

    if not os.path.exists(proto_path) or not os.path.exists(weights_path):
        print("Downloading HED model (~100MB total)...")
        try:
            urllib.request.urlretrieve(proto_url, proto_path)
            urllib.request.urlretrieve(weights_url, weights_path)
        except Exception as e:
            print(f"Failed to download HED model: {e}")
            return None

    print("Loading HED model...")
    try:
        net = cv2.dnn.readNetFromCaffe(proto_path, weights_path)
        return net
    except Exception as e:
        print(f"Failed to load HED model: {e}")
        return None
    

hed_net = load_hed_model() if EDGE_METHOD == "hed" else None
if EDGE_METHOD == "hed" and hed_net is None:
    print("⚠️ Falling back to Canny edge detection because HED failed to load.")
    EDGE_METHOD = "canny"


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
