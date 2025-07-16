#Mean average precition
from collections import Counter
import torch
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn as nn
def compute_iou(box1, box2):
    """
    Compute IoU between two boxes (x, y, w, h format)
    box1, box2: shape (..., 4) where 4 = (xc, yc, w, h)
    Returns: IoU (..., 1)
    """
    b1_x1 = box1[..., 0] - box1[..., 2] / 2
    b1_y1 = box1[..., 1] - box1[..., 3] / 2
    b1_x2 = box1[..., 0] + box1[..., 2] / 2
    b1_y2 = box1[..., 1] + box1[..., 3] / 2

    b2_x1 = box2[..., 0] - box2[..., 2] / 2
    b2_y1 = box2[..., 1] - box2[..., 3] / 2
    b2_x2 = box2[..., 0] + box2[..., 2] / 2
    b2_y2 = box2[..., 1] + box2[..., 3] / 2

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union = area1 + area2 - inter_area

    iou = torch.where(
        union > 0,
        inter_area / union,
        torch.zeros_like(union)
    )
    return iou


class YoloV1Loss(nn.Module):
    def __init__(self, S=8, B=2, C=2, lambda_coord = 5.0, lambda_noobj = 0.5):
        super(YoloV1Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse = nn.MSELoss(reduction="sum") 

    def forward(self, predictions, target):
        """
        predictions: (batch_size, S,S, B*5 + C)
        target: (batch_size, S,S, B*5 + C)
        """
        N = predictions.size(0) #batch size

        #split predictions into components
        pred_boxes = predictions[..., :self.B*5].clone() # shape: (N, S, S, 10) for B = 2
        pred_classes = predictions[..., self.B*5:] # shape:(N, S, S, C)

        #similarly split target 
        target_boxes = target[..., :self.B*5]
        target_classes = target[..., self.B*5:]

        # For each grid cell, is there an object?
        obj_mask = target_boxes[..., 0::5].sum(dim=-1, keepdim=True) > 0
        # [..., 0::5] picks up values at positions which are of the multiples of 5. So it takes the confidence values of each box (N,S,S,2)
        # Then it is summed to get (N,S,S,1)
        # If their sum is non-zero then 1. That means there contains an object in the grid
        obj_mask = obj_mask.float() # Converts mask which is a boolean tensor to float for computation
        noobj_mask = 1 - obj_mask
        
        target_xywh = target_boxes[..., 1:5]
        ious = []
        for b in range(self.B):
            pred_xywh = pred_boxes[..., b*5+1:b*5+5]
            iou = compute_iou(pred_xywh, target_xywh)
            ious.append(iou.unsqueeze(-1))

        ious = torch.cat(ious, dim=-1) #concantate the two IOUs # shape: (N, S, S, B)
        best_box = torch.argmax(ious, dim=-1, keepdim=True) # # shape: (N, S, S, 1)

        # compute localization loss
        coord_loss = 0.0

        for b in range(self.B):
            mask = (best_box == b).float() * obj_mask  # (N, S, S, 1)

            pred_xy = pred_boxes[..., b*5+1 : b*5+3]
            pred_wh = pred_boxes[..., b*5+3 : b*5+5]
            target_xy = target_boxes[..., 1:3]
            target_wh = target_boxes[..., 3:5]

            pred_wh_sqrt = torch.sign(pred_wh) * torch.sqrt(torch.abs(pred_wh + 1e-6))
            target_wh_sqrt = torch.sqrt(target_wh)

            coord_loss += self.mse(mask * pred_xy, mask * target_xy) + \
                          self.mse(mask * pred_wh_sqrt, mask * target_wh_sqrt)

        coord_loss *= self.lambda_coord

        # 3. Compute Confidence Loss
        obj_conf_loss = 0.0
        noobj_conf_loss = 0.0

        for b in range(self.B):
            pred_conf = pred_boxes[..., b*5+0].unsqueeze(-1)
            target_conf = target_boxes[..., 0].unsqueeze(-1)  # Always 1 where obj exists

            mask = (best_box == b).float() * obj_mask  # object presence + best box

            obj_conf_loss += self.mse(mask * pred_conf, mask * target_conf)
            noobj_conf_loss += self.mse(noobj_mask * pred_conf, noobj_mask * target_conf)

        noobj_conf_loss *= self.lambda_noobj

        # 4. Classification Loss (only if object is present)
        class_loss = self.mse(obj_mask * pred_classes, obj_mask * target_classes)

        # 5. Total Loss
        total_loss = (coord_loss + obj_conf_loss + noobj_conf_loss + class_loss) / N
        return total_loss



def mean_average_precition(pred_boxes, target_boxes, IoU_thresh = 0.5, C=2):
    """
    pred_boxes (list): list of lists containing all bboxes with each bboxes
    specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
    target_boxes (list): same as the above but for ground truth boxes
    IoU_thresh (float): The threshold IoU for prediction to be considered as accurate
    C (int): number of classes
    """
    average_precisions = [] #List containing AP scores for all classes
    eps = 1e-6

    for c in range(C):
        detections = []
        ground_truths = []
        
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for target in target_boxes:
            if target[1] == c:
                ground_truths.append(target)

        amount_bbox = Counter([gt[0] for gt in ground_truths]) #gt[0] is the idx of the ground truth box
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bbox.items():
            amount_bbox[key] = torch.zeros(val)

        #sort by box probabilities
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue
        
        #Only take the ground truths that has the same training idx as detection
        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]
            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = compute_iou(
                    torch.tensor(detection[3:]), #0:2 is idx class and probability\
                    torch.tensor(gt[3:])
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > IoU_thresh:
                # only detect ground truth detection once
                # We use the amount_bboxes dictionary to keep track of the usage of gt boxes
                if amount_bbox[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 0
                    amount_bbox[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
        
        TP_cumsum = torch.cumsum(TP, dim = 0)
        FP_cumsum = torch.cumsum(FP, dim = 0)
        recalls = (TP_cumsum) / (total_true_bboxes + eps)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + eps))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

def non_max_suppression(bboxes, iou_threshold, threshold):
    """
    Does Non Max Suppression given bboxes
    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes
    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or compute_iou(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def plot_loss(train_loss, val_loss, epochs, mAP=None, save_path="training_loss.png"):
    """
    Plots training and validation loss on a single graph, and optionally mAP on a secondary y-axis.

    Args:
        train_loss (list or np.ndarray): Training loss per epoch.
        val_loss (list or np.ndarray): Validation loss per epoch.
        epochs (int): Number of epochs.
        mAP (list or np.ndarray, optional): mAP per epoch.
        save_path (str): Path to save the plot image.
    """
    epochs_range = np.arange(epochs)
    fig, ax1 = plt.subplots(figsize=(9, 6))

    # Plot Losses
    ax1.plot(epochs_range, train_loss, 'r--', label='Train Loss')
    ax1.plot(epochs_range, val_loss, 'g-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training/Validation Loss' + (' and mAP' if mAP is not None else ''))
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Plot mAP if provided
    if mAP is not None:
        ax2 = ax1.twinx()
        ax2.plot(epochs_range, mAP, 'b-o', label='mAP')
        ax2.set_ylabel('mAP')
        ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
            )


            #if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes



def convert_cellboxes(predictions, S=8, B=2, C=2):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, B * 5 + C)
    bboxes1 = predictions[..., C + 1:C + 5]
    bboxes2 = predictions[..., C + 6:C + 10]
    scores = torch.cat(
        (predictions[..., C].unsqueeze(0), predictions[..., C + 5].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :C].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., C], predictions[..., C + 5]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=8, B=2, C=2):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


