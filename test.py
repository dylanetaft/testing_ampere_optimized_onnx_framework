import os
import urllib.request
import zipfile
import glob
import cv2
import numpy as np
import onnxruntime as ort
import time
import threading

def ensure_coco_dataset():
    coco_dir = "./coco128"
    zip_path = "coco128.zip"
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"

    # Check if dataset directory already exists
    if os.path.isdir(coco_dir):
        print(f"Directory '{coco_dir}' already exists. Skipping download.")
        return

    print(f"Directory '{coco_dir}' not found. Downloading dataset...")

    # Download the zip file
    urllib.request.urlretrieve(url, zip_path)
    print(f"Downloaded {zip_path}")

    # Unzip contents
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")  # Extract into current directory
    print(f"Extracted {zip_path}")

    # Optionally remove the zip file
    os.remove(zip_path)
    print(f"Removed {zip_path}")

def letterbox_image(image, target_size=(640, 640), color=(114, 114, 114)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    nh, nw = int(h * scale), int(w * scale)

    # Resize keeping aspect ratio
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Compute padding
    pad_h = target_size[0] - nh
    pad_w = target_size[1] - nw
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # Add padding
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded

def infer(session, img):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    inputs = { input_name: img }
    return session.run([output_name], inputs)

def prepare_images(path, target_size=(640, 640)):
    jpg_files = glob.glob(os.path.join(path, "*.jpg"))
    processed_images = []

    for file in jpg_files:
        img = cv2.imread(file)
        if img is None:
            print(f"Failed to load {file}")
            continue
        img = letterbox_image(img, target_size)
        img = img.transpose(2,0,1) #hwc->chw
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        processed_images.append(img)
    
    return processed_images



ensure_coco_dataset()
images = prepare_images('./coco128/images/train2017')

sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 32 
sess_options.inter_op_num_threads = 1 
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession("yolov8s-worldv2.onnx", sess_options, providers=ort.get_available_providers())
outer_threads = 1 

while True:
    threads = []
    start_time = int(time.time() * 1000)
    for t_index in range(outer_threads):
        t = threading.Thread(target=infer, args=(session, images[t_index]))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    end_time = int(time.time() * 1000)

    print ((end_time - start_time)/outer_threads)
