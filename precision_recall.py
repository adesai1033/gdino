
from collections import Counter
from run import iou
import os
from torchvision.ops import box_convert
import cv2
import torch
import numpy as np
from random import shuffle
def apr(ground_truth_boxes, detected_boxes, iou_threshold=0.5):
    """
    Calculate the mean average precision for a set of ground truth boxes and detected boxes.
    """
    # Initialize variables

    true_positives = 0
    false_positives = 0
    
    # Sort detected boxes by confidence score in descending order
    #detected_boxes.sort(key=lambda x: x[4], reverse=True)
    
    # Keep track of which ground truth boxes have been detected
    detected_counter = [False] * len(ground_truth_boxes)
    
    # Loop over detected boxes
    for detected_box in detected_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        # Loop over ground truth boxes
        for gt_idx in range(len(ground_truth_boxes)):
            #currIou = iou(detected_box[:-1], ground_truth_boxes[gt_idx])
            currIou = iou(detected_box, ground_truth_boxes[gt_idx])
            
            # Update best IoU and index if this IoU is better and the ground truth box hasn't been detected yet
            if currIou > best_iou and not detected_counter[gt_idx]:
                best_iou = currIou
                best_gt_idx = gt_idx
        
        # If the best IoU is above the threshold, it's a true positive
        if best_iou >= iou_threshold:
            true_positives += 1
            detected_counter[best_gt_idx] = True
            
        else:
            false_positives += 1
            
    false_negatives = len(ground_truth_boxes) - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall

def mapr(ground_truth_boxes, detected_boxes, threshold, step):

    precisions = []
    recalls = []

    while threshold < 1:
        precision, recall = apr(ground_truth_boxes, detected_boxes, threshold)
        precisions.append(precision)
        recalls.append(recall)
        threshold += step

    return sum(precisions)/len(precisions), sum(recalls)/len(recalls)


def yolo_to_xyxy(yolo_bbox):
    cx, cy, w, h = yolo_bbox[1], yolo_bbox[2], yolo_bbox[3], yolo_bbox[4]
    topleft_x = cx - w/2
    topleft_y = cy - h/2
    btmright_x = cx + w/2
    btmright_y = cy + h/2
    return [topleft_x, topleft_y, btmright_x, btmright_y]

def gtfile_to_list(textfile):
    boxes_lst = []
    with open(textfile, "r") as f:
        lines = f.readlines()
        f.close()
    for line in lines:
        box = line.split(' ')
        floatbox = [float(x) for x in box]
        boxes_lst.append(floatbox[1:])
    return boxes_lst
def mfile_to_list(textfile):
    boxes_lst = []
    with open(textfile, "r") as f:
        lines = f.readlines()
        f.close()
    for line in lines:
        box = line.split(' ')
        floatbox = [float(x) for x in box]
        tensorbox = torch.Tensor(floatbox)
        xyxy = list(tensorbox)
        boxes_lst.append(xyxy)
    return boxes_lst

def main():
    GTBOXES_INPUT_DIR = '/mnt/bpd_architecture/abdesa/ground_truth_text/'  #in yolo format 
    MBOXES_INPUT_DIR = '/mnt/bpd_architecture/abdesa/dino_accuracy_text/' #path to directory containing textfile outputs from dino (in xyxy format)
    IMAGES_INPUT_DIR = '/mnt/bpd_architecture/abdesa/dino_accuracy_images/'
    
    directory_precision = []
    directory_recall = []
    filenum = 0
    for gtfile, mfile in zip(os.listdir(GTBOXES_INPUT_DIR), os.listdir(MBOXES_INPUT_DIR)): 
        gtfile = os.path.join(GTBOXES_INPUT_DIR, gtfile)
        mfile = os.path.join(MBOXES_INPUT_DIR, mfile)
        #imgfile = os.path.join(IMAGES_INPUT_DIR, imgfile)
        #image = cv2.imread(imgfile)
        #image_np = np.array(image)
        gtboxes_lst = gtfile_to_list(gtfile)
        mboxes_lst = mfile_to_list(mfile)
        
        #print(gtboxes_lst)
        #print('###############################################################')
        #print(mboxes_lst)
        #shuffle(gtboxes_lst)
        #map(shuffle, gtboxes_lst)
        
        file_precision, file_recall = apr(gtboxes_lst, mboxes_lst)
        directory_precision.append(file_precision)
        directory_recall.append(file_recall)
        print(f'File {filenum} processed')
        filenum += 1
    print(f'Avg Precision: {sum(directory_precision)/len(directory_precision)}\nAvg Recall: {sum(directory_recall)/len(directory_recall)}\n')
    print(f'Best Precision: {max(directory_precision)}\nBest Recall: {max(directory_precision)}')

if __name__ == '__main__':
    main()