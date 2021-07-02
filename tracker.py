from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
import numpy
import random
import os
import torchvision.transforms as transforms
from PIL import Image
from myDataset import judgement, ReID_Net
from myLoad import Load
model, device = Load()
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)
point_res = numpy.zeros((50,100)).tolist()
def trail_res(image, pos_id, c3):
    color = ((int)(255-5*pos_id),(int)(5*pos_id),0)
    point_res[pos_id].append(c3)
    for points in point_res[pos_id]:
        cv2.circle(image, points, 1, color, thickness=0)

point_list = []
_color = []
def random_color():
    for i in range(50):
        _color.append((random.randint(0,255),random.randint(0,255),random.randint(0,255)))
def plot_bboxes(image, bboxes, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    for (pos_id, x1, y1, x2, y2, cls_id) in bboxes:
        cut = image[y1:y2, x1:x2]
        mean = [0.5071, 0.4867, 0.4408]
        stdv = [0.2675, 0.2565, 0.2761]
        judge_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv)])
        judge_path = './judge'
        if (os.path.exists(judge_path)):
            cv2.imwrite(judge_path + '/' + '1.jpg', cut)
        else:
            os.makedirs(judge_path)
            cv2.imwrite(judge_path + '/' + '1.jpg', cut)
        cut = Image.open('./judge/1.jpg').convert('RGB')
        judge = judge_transforms(cut)
        print(judge.size())
        judge = judge.reshape(1, 3, 256, 256)
        print(judge.size())
        predict = judgement(model, judge, device)
        print(predict[0].item())

        if cls_id in ['smoke', 'phone', 'eat']:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        if cls_id == 'eat':
            cls_id = 'eat-drink'
        # draw(pos_id, x1, x2, y1, y2, image)
        c1, c2 = (x1, y1), (x2, y2)
        x3 = (int)(0.5 * (x1 + x2))
        y3 = (int)(0.5 * (y1 + y2))
        # c3 = (pos_id, x3, y3)
        c3 = (predict[0].item(),x3,y3)
        # c3 = filter(x1, x2, y1, y2)
        # trail_res(image, pos_id, c3)
        point_list.append(c3)
        # for points in point_list:
        #     cv2.circle(image, points, 1, color, thickness=0)
        # cv2.circle(image, c3, 1, color, thickness=0)
        # cv2.circle(image,c3,1,color,thickness=8)

        cv2.rectangle(image, c1, c2, _color[predict[0].item()], thickness=tl, lineType=cv2.LINE_AA)
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, _color[predict[0].item()], -1, cv2.LINE_AA)  # filled
        cv2.putText(image, '{} ID-{}'.format(cls_id, predict[0].item()), (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

        # path = './data/train/' + str(pos_id)
        # if(os.path.exists(path)):
        #     cv2.imwrite(path + '/' + str(x1) + '.jpg', cut)
        # else:
        #     os.makedirs(path)
        #     cv2.imwrite(path + '/' + str(x1) + '.jpg', cut)
    return image


def update_tracker(target_detector, image):

        new_faces = []
        _, bboxes = target_detector.detect(image)
        print(bboxes)
        print("-----")
        bbox_xywh = []
        confs = []
        bboxes2draw = []
        face_bboxes = []
        if len(bboxes):

            # Adapt detections to deep sort input format
            for x1, y1, x2, y2, _, conf in bboxes:
                                                        #x,y of center point, width, height
                obj = [
                    int((x1+x2)/2), int((y1+y2)/2),
                    x2-x1, y2-y1
                ]
                bbox_xywh.append(obj)
                confs.append(conf)

            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)

            # Pass detections to deepsort
            outputs = deepsort.update(xywhs, confss, image)
            # print(outputs)
            for value in list(outputs):
                x1,y1,x2,y2,track_id = value
                bboxes2draw.append(
                    (track_id, x1, y1, x2, y2, '')
                )

        random_color()
        image = plot_bboxes(image, bboxes2draw)

        return image, new_faces, face_bboxes
