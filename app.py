from flask import Flask, request, render_template
import os
import cv2
import requests
from matplotlib import rcParams
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
from io import BytesIO
import base64

# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog


from detectron2.utils.visualizer import ColorMode
import glob


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

#test evaluation
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
#cfg.DATASETS.TRAIN = ("my_dataset_train",)
#cfg.DATASETS.TEST = ("my_dataset_val",)

cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.001


cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1500 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = (1000, 1100)
cfg.SOLVER.GAMMA = 0.05




cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5 #your number of classes + 1

cfg.TEST.EVAL_PERIOD = 500


# Directly provide the full path to the pretrained model weights
cfg.MODEL.WEIGHTS = "model_final.pth"

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
predictor = DefaultPredictor(cfg)


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def process_image(image_path):
    im = cv2.imread(image_path)
    outputs = predictor(im)
    
    v = Visualizer(im[:, :, ::-1], scale=0.8)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Save the processed image
    processed_image_path = 'static/Images/abc.png'
    out1 = Image.fromarray(out.get_image()[:, :, ::-1])
    out1.save(processed_image_path)
    
    # Extract bounding box coordinates
    bounding_box_coordinates = []
    for bbox in outputs["instances"].pred_boxes.tensor:
        x, y, w, h = map(int, bbox)
        bounding_box_coordinates.append(((x, y), (x + w, y + h)))

    return processed_image_path, bounding_box_coordinates


@app.route('/')
def upload_form():
    return render_template('abc.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = file.filename
        file_path = filename
        file.save(file_path)
        processed_image_path, bounding_boxes = process_image(file_path)
        return render_template('results.html', image_path=processed_image_path, bounding_boxes=bounding_boxes)

"""
if __name__ == "__main__":
    app.run()
"""
        
@app.route('/sai')
def sai():
    return 'Hi sai'

if __name__ == '__main__':
    app.run(debug=True, port=8004)
