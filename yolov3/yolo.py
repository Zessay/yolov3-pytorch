import os
import torch
import subprocess
from yolov3.models import Darknet
from yolov3.utils.utils import non_max_suppression, rescale_boxes
from yolov3.utils.datasets import pad_to_square, resize

def download_url(url, outdir):
    print(f'Downloading files from {url}')
    cmd = ['wget', '-c', url, '-P', outdir]
    subprocess.call(cmd)


class YOLOv3:
    def __init__(
            self,
            yolo_weights_file: str,
            yolo_cfg_file: str,
            device,
            img_size=416,
            person_detector=False,
            video=False,
            return_dict=False
    ):
        if not os.path.isfile(yolo_weights_file):
            os.makedirs(os.path.dirname(yolo_weights_file), exist_ok=True)
            url = 'https://pjreddie.com/media/files/yolov3.weights'
            outdir = os.path.dirname(yolo_weights_file)
            download_url(url, outdir)

        if not os.path.isfile(yolo_cfg_file):
            os.makedirs(os.path.dirname(yolo_cfg_file), exist_ok=True)
            url = 'https://raw.githubusercontent.com/mkocabas/yolov3-pytorch/master/yolov3/config/yolov3.cfg'
            outdir = os.path.dirname(yolo_cfg_file)
            download_url(url, outdir)

        self.conf_thres = 0.8
        self.nms_thres = 0.4
        self.img_size = img_size
        self.video = video
        self.person_detector = person_detector
        self.device = device
        self.return_dict = return_dict

        self.model = Darknet(yolo_cfg_file, img_size=img_size).to(device)
        self.model.load_darknet_weights(yolo_weights_file)
        # self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

    @torch.no_grad()
    def __call__(self, batch):
        if self.video:
            inp_batch = []
            for img in batch:
                # Pad to square resolution
                img, _ = pad_to_square(img, 0)
                # Resize
                img = resize(img, self.img_size)
                inp_batch.append(img)
            inp_batch = torch.stack(inp_batch).float().to(self.device)
        else:
            inp_batch = batch

        detections = self.model(inp_batch)
        detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)

        for idx, det in enumerate(detections):
            if det is None:
                det = {
                    'boxes': torch.empty(0,4),
                    'scores': torch.empty(0),
                    'classes': torch.empty(0),
                }
                detections[idx] = det
                continue

            if self.video:
                det = rescale_boxes(det, self.img_size, batch.shape[-2:])

            if self.person_detector:
                det = det[det[:,6] == 0]

            if self.return_dict:
                det = {
                    'boxes': det[:, :4],
                    'scores': det[:, 4] * det[:, 5],
                    'classes': det[:, 6],
                }

            detections[idx] = det



        return detections
