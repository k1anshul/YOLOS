import torch
import torchvision
import os
import cv2
import yaml
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
from shapely.geometry import box
from transformers import AutoFeatureExtractor
from transformers import DetrConfig, AutoModelForObjectDetection
import pytorch_lightning as pl


# Loading inference config file
parser = argparse.ArgumentParser()
parser.add_argument('--hyp', type=str, default=r"config/hyp_inference.yaml",
                    help="path of inference config file")  # parameter to add hyp.yaml file
opt = parser.parse_args()
hyp = yaml.load(open(opt.hyp), Loader=yaml.FullLoader)

feature_extractor = AutoFeatureExtractor.from_pretrained(
    hyp['model_name'], size=hyp['imgz'])

id2label={k:v for k,v in enumerate(hyp['id2label'])}

class CocoDetection(torchvision.datasets.CocoDetection):
    pass


def collate_fn(batch):
    pass

# we wrap our model around pytorch lightning for training


class YoloS(pl.LightningModule):

    def __init__(self, lr, weight_decay):
        super().__init__()
        # replace COCO classification head with custom head
        self.model = AutoModelForObjectDetection.from_pretrained(
             hyp['model_name'], num_labels=len(id2label), ignore_mismatched_sizes=True)
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()  # adding this will save the hyperparameters to W&B too

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)

        return outputs


model = torch.load(hyp['model_path'])

class Colors:
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2bgr('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2bgr(h): 
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (4, 2, 0))

colors = Colors()

def box_iou(box1, box2):
    box1 = box(box1[0], box1[1], box1[2], box1[3])
    box2 = box(box2[0], box2[1], box2[2], box2[3])

    # iou = inter / (area1 + area2 - inter)
    return (box1.intersection(box2).area)/(box1.union(box2).area)


def suppression(prob1, box1, probs, boxes, iou_thres=0.45):
    for prob2, box2 in zip(probs, boxes.tolist()):
        if (box2 != box1) and (prob2.argmax() == prob1.argmax()):
            if box_iou(box1, box2) >= iou_thres:
                if prob1[prob1.argmax()] < prob2[prob2.argmax()]:
                    return True


# for output bounding box post-processing

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size[1], size[0]
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(pil_img, prob, boxes, output_fld, img, iou):

    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):

        box = (xmin, ymin, xmax, ymax)

        if not suppression(p, box, prob, boxes, iou_thres=iou):

            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            cl = p.argmax()  # class index
            # class_name and confidenece
            text = f'{id2label[cl.item()]}: {p[cl]:0.2f}'
            cv2.rectangle(pil_img, (xmin, ymin), (xmax, ymax), colors(cl), 3)
            cv2.putText(pil_img, text, (xmin, ymin),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, colors(cl), 3, cv2.LINE_AA)
            cv2.imwrite(os.path.join(output_fld, img), pil_img)


def visualize_predictions(image, outputs, output_fld, img, conf, iou):
    # keep only predictions with confidence >= threshold
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > conf

    # convert predicted boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(
        outputs.pred_boxes[0, keep].cpu(), image.shape)

    # plot results
    plot_results(image, probas[keep], bboxes_scaled, output_fld, img, iou)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
os.makedirs(hyp['output_fld'], exist_ok=True)

for img in tqdm(os.listdir(hyp['test_data'])):
    if img.endswith('.jpg'):
        image=cv2.imread(os.path.join(hyp['test_data'], img))
        outputs = model(pixel_values=feature_extractor(images=image, return_tensors="pt")["pixel_values"].cuda())
        visualize_predictions(image, outputs, hyp['output_fld'], img, hyp['confidenec'], hyp['iou'])