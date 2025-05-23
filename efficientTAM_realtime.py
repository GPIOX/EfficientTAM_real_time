import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from pathlib import Path
# from extract_main_subject import extract_main_subject_with_category
from datasets import build_dataset
import argparse
import torch
import os
from torchvision import transforms
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import pandas as pd
from davis2017.evaluation import DAVISEvaluation
from datasets.transform_utils import VideoEvalDataset
from torch.utils.data import DataLoader
import util.misc as utils
from os.path import join
import cv2

from efficient_track_anything.build_efficienttam import (
    build_efficienttam_camera_predictor,
)
from PIL import Image
from ultralytics import YOLO

def save_image(output_dir, multi_target_output_dict, idx, palette, current_obj_nums):
    out_mask_logits = multi_target_output_dict[0]["pred_masks"]

    save_img = np.zeros((out_mask_logits.shape[-2], out_mask_logits.shape[-1]))
    for i in range(current_obj_nums):
        out_mask = (multi_target_output_dict[i]["pred_masks"] > 0.0).cpu().numpy()
        save_img[out_mask.squeeze()] = i+1

    img_E = Image.fromarray(save_img.astype(np.uint8))
    img_E.putpalette(palette)
    # img_E.save(f'output/MyData/result_{idx}.png')
    img_E.save(join(output_dir, f'{idx}.png'))

def main(args):
    checkpoint = "EfficientTAM/checkpoints/efficienttam_s.pt"
    model_cfg = "configs/efficienttam/efficienttam_s.yaml"
    predictor = build_efficienttam_camera_predictor(model_cfg, checkpoint, device="cuda")

    cap = cv2.VideoCapture(args.video_path)

    if_init = False
    palette_img = os.path.join('data/ref-davis', "valid/Annotations/blackswan/00000.png")
    palette = Image.open(palette_img).getpalette()
    idx=0
    yolo_world = YOLO("yolov8x-world.pt")
    yolo_world.set_classes(['ship'])
    current_obj_nums = 0
    obj_id_list=[]

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            width, height = frame.shape[:2][::-1]
            if not if_init:
                predictor.load_first_frame(frame)
                if_init = True
            # if not det_flag:
            results = yolo_world.predict(
                source=frame,
                verbose=False,
                conf=0.5,
            )
            if results[0].boxes.xyxy.shape[0] == 0:
                idx += 1
                predictor.load_other_frame(frame)
            else:
                if results[0].boxes.xyxy.shape[0] > current_obj_nums:
                    current_obj_nums = results[0].boxes.xyxy.shape[0]

                    ann_obj_id = tuple(range(results[0].boxes[-1].xyxy.shape[0]))
                    # 获取YOLO-World的检测结果
                    box = results[0].boxes[-1].xyxy.cpu().numpy()
                    box = box.reshape(-1, 2)

                    # if not if_init:
                    #     predictor.load_first_frame(frame)
                    #     if_init = True
                    predictor.load_other_frame(frame)
                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(frame_idx=idx, obj_id=current_obj_nums-1, box=box)
                    obj_id_list.append(current_obj_nums-1)
                    idx += 1
                else:
                    multi_target_output_dict = predictor.track_multi_target(frame, obj_id_list)
                    predictor.load_other_frame(frame)
                    # print(current_obj_nums, multi_target_output_dict.keys())
                    save_image(args.output_dir, multi_target_output_dict, idx, palette, current_obj_nums)
                    idx += 1
            print("Current frame processed: ", idx)

            # results[0].save(f'output/MyData/result_{idx}.png')
            # idx += 1
            # if not if_init:
            #     predictor.load_first_frame(frame)
            #     if_init = True
            #     _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(frame_idx=0, obj_id=0, box=np.array([284, 201, 486, 218]))
            #     idx = 0
            # else:
            #     out_obj_ids, out_mask_logits = predictor.track(frame)
            #     out_mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            #     save_img = np.zeros((out_mask_logits.shape[-2], out_mask_logits.shape[-1]))
            #     img_E = Image.fromarray(save_img.astype(np.uint8))
            #     img_E.putpalette(palette)
            #     img_E.save(f'output/MyData/result_{idx}.png')
            #     idx += 1
                # print('ok')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('EfficientTAM demo script', add_help=False)
    parser.add_argument('--video-path', default='MyData/无标题视频——使用Clipchamp制作 (1).mp4', type=str)
    parser.add_argument('--output-dir', default='output/MyData', type=str)
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.video_path.split('/')[-1].split('.')[0])
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)