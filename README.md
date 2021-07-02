# 注意：

本项目使用Yolov5 3.0版本，4.0版本需要替换掉models和utils文件夹

# 项目简介：
使用YOLOv5+Deepsort+VGG实现行人MOT和ReID。

代码地址（欢迎star）：

[https://github.com/SekiroRong/yolo-vgg-MOT](https://github.com/SekiroRong/yolo-vgg-MOT)

最终效果：
![QQ图片20210702130147.png](https://i.loli.net/2021/07/02/DeyhK3cvtJR6kNY.png)
# YOLOv5检测器：

```python
class Detector(baseDet):

    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()
        self.build_config()

    def init_model(self):

        self.weights = 'weights/yolov5m.pt'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        model = attempt_load(self.weights, map_location=self.device)
        model.to(self.device).eval()
        model.half()
        # torch.save(model, 'test.pt')
        self.m = model
        self.names = model.module.names if hasattr(
            model, 'module') else model.names

    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()  # 半精度
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def detect(self, im):

        im0, img = self.preprocess(im)

        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.4)

        pred_boxes = []
        for det in pred:

            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if not lbl in ['person', 'car', 'truck']:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return im, pred_boxes

```

调用 self.detect 方法返回图像和预测结果

# DeepSort追踪器：

```python
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)
```

调用 self.update 方法更新追踪结果

# 利用DeepSort获取视频数据集
```
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
```
# 运行myDataset训练模型：

```
class ReID_Net(nn.Module):
    def __init__(self):
        super(ReID_Net, self).__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size=7,dilation=2)
        self.conv2 = nn.Conv2d(16,32, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(55696*2,10000)
        self.fc2 = nn.Linear(10000, 1000)
        self.fc3 = nn.Linear(1000, 100)
        self.fc4 = nn.Linear(100, 35)
    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1,55696*2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = F.relu(self.fc3(x))
        x = F.dropout(x)
        x = self.fc4(x)
        return F.log_softmax(x,dim=1)
```
训练好后放到 weights 文件夹下

# 运行demo生成result.mp4：

# 联系作者：

> Github：[https://github.com/SekiroRong](https://github.com/SekiroRong)
