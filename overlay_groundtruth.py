from groundingdino.util.inference import load_model, load_image, predict, annotate
import os
from torchvision.ops import box_convert
import cv2
from precision_recall import mfile_to_list
import torch
import numpy as np

def overlay_boxes(image, gt_boxes, filenum, path, imgname):
    for gt_box in gt_boxes:
        height, width, channels = image.shape
        #gt_box_tensor = torch.Tensor(gt_box)
        #xyxy = list(box_convert(boxes=gt_box_tensor, in_fmt="cxcywh", out_fmt="xyxy").numpy())
        '''
        cx, cy, w, h = gt_box[0], gt_box[1], gt_box[2], gt_box[3],
        topleft_x = cx - w/2
        topleft_y = cy - h/2
        btmright_x = cx + w/2
        btmright_y = cy + h/2
        xyxy = [topleft_x, topleft_y, btmright_x, btmright_y]
        start_x, start_y, end_x, end_y = int(xyxy[0] * width) , int(xyxy[1] * height), int(xyxy[2] * width), int(xyxy[3] * height)
       '''
        #start_x, start_y, end_x, end_y = int(xyxy[0] ), int(xyxy[1]  ), int(xyxy[2] ), int(xyxy[3])
        cv2.rectangle(
            image,
            (int(gt_box[0] * width), int(gt_box[1] * height)),
            (int(gt_box[2] * width), int(gt_box[3] * height)),
            (255, 0, 0),
            1
        )
        
    
        cv2.imwrite(os.path.join(path, imgname), image)
def main():
    
    IMAGE_DIR = '/mnt/bpd_architecture/abdesa/COCO_overlaid'
    GT_TXT_DIR = '/mnt/bpd_architecture/abdesa/dino_accuracy_text'
    OVERLAID_DIR =  '/mnt/bpd_architecture/abdesa/samCoco_format_test/'
    PARENT_DIRECTORY = '/mnt/bpd_architecture/abdesa/'
    os.mkdir(os.path.join(PARENT_DIRECTORY, OVERLAID_DIR))

    filenum = 0
    for txt, img in zip(os.listdir(GT_TXT_DIR), os.listdir(IMAGE_DIR)): #fixlater
        #read file
        imgname = img
        txt_path = os.path.join(GT_TXT_DIR, txt)
        image_path = os.path.join(IMAGE_DIR, img)
        
        image = cv2.imread(image_path)
        gtboxes_lst = mfile_to_list(txt_path)
        overlay_boxes(image, gtboxes_lst, filenum, OVERLAID_DIR, imgname)

        print("filenum:", filenum)
        filenum += 1

if __name__ == '__main__':
    main()