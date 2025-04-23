import sys
sys.path.append('./lib/cad')
import lib.third_party.face_recog.face_recog as face_recog
import lib.cad.cad_detector as cad_detector
import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import os
from PIL import Image
import re
import supervision as sv
import argparse

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness = 5)
LABEL_ANNOTATOR = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_scale=0.5, text_thickness=1)

def to_video(video_log_dir):
    image_folder = video_log_dir
    video_name = f'{video_log_dir}/anno.mp4'
    images = [img for img in os.listdir(image_folder) if img.endswith("anno.png")]
    print(images)
    images.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))  # 按数字排序
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 适用于mp4格式
    video = cv2.VideoWriter(video_name, fourcc, 30.0, (width, height))
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)
    video.release()
    print(f"Video saved as {video_name}")

def annotate_cv2_frame(frame, detections, labels, cad_threshold = False):
    image = frame
    if detections:
        if cad_threshold:
            for i in range(len(labels)):
                detection = detections[i]
                label = labels[i]
                cad_prob = float([_ for _ in label.split(',') if 'cad' in _][1].split('@')[-1])

                if cad_prob > cad_threshold:
                    color = sv.Color.RED
                else:
                    color = sv.Color.BLUE
                BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(color=color, thickness=5)
                image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections=detection)
        else:
            image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
        image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    return image

class DigitalShadow:
    def __init__(self, livecad_model = None, recog = None, recog_ref_path=None):
        self.model_cad = cad_detector.cad_detector_vit(
            diff_model_root = livecad_model
        )
        self.model_face_recog = face_recog.FaceDetector(recognition=recog, user_face_reference_img_path=recog_ref_path, tolerance = 0.6)
        self.risks = []
        self.risks_buffer = 20
        
    def cad_predict_one_cv2_frame(self, frame, show = True, log_dir = None, log_id = None):
        ori_frame = deepcopy(frame)
        _, detections, texts, labels, crops = self.model_face_recog.process_one_cv2_frame(frame = frame, extend_ratio = 1.1)
        print(detections, labels, crops)
        
        cad_risk = None
        for idx, crop in enumerate(crops):
            if show:
                plt.imshow(crop)
                plt.show()
                plt.close()
            crop_cv_bgr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
            predictions = self.model_cad.predict_single_cv2_frame(frame = crop_cv_bgr)
            cad_risk = round(predictions[0][1],3)
            if len(self.risks) < self.risks_buffer:
                self.risks.append(cad_risk)
            else:
                self.risks.pop(0)
                self.risks.append(cad_risk)
            avg_risk = np.mean(self.risks)
            labels[idx] += f', cad risk (real time) {predictions[0][0]}@{round(predictions[0][1],3)}, cad risk (avg) {predictions[0][0]}@{round(avg_risk,3)}'
            
        frame = annotate_cv2_frame(frame, detections, labels, cad_threshold = 0.4)
            
        annotate_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotate_frame_rgb = Image.fromarray(annotate_frame_rgb)
        ori_frame_rgb = cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB)
        ori_frame_rgb = Image.fromarray(ori_frame_rgb) 
        
        if log_dir:
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            annotate_frame_rgb.save(f'{log_dir}/{log_id}.anno.png')
            ori_frame_rgb.save(f'{log_dir}/{log_id}.ori.png')
            
        return annotate_frame_rgb, cad_risk
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="DigitalShadow", add_help=False)
    parser.add_argument('--video',
                        help='path/to/video',
                        default='./test_data/crowd.mp4')
    parser.add_argument('--livecad_model',
                        help='path/to/livecad/model',
                        default='./model/cad_beitv2_face_20240905')
    parser.add_argument('--recog',
                        help='perform face recognition or not',
                        default=False)
    parser.add_argument('--recog_ref_path',
                        help='path to face reference for face recognition',
                        default='./user_profile')
    args = parser.parse_args()
    
    DS = DigitalShadow(livecad_model = args.livecad_model, recog = args.recog, recog_ref_path = args.recog_ref_path)
    
    if True:
        video_path = args.video
        video_log_dir = video_path + '.log'
        os.makedirs(video_log_dir, exist_ok=True)
        print(video_path)

        try:
            # 打开摄像头
            cap = cv2.VideoCapture(int(float(video_path)))
            print(f'open camera {video_path}')
        except:
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            print(f'open {video_path}')

        count = 0
        fps = 1

        file = open(f'{video_log_dir}/cad_risk.txt', 'w')
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if count % fps == 0:
                    annotate_frame_rgb, cad_risk = DS.cad_predict_one_cv2_frame(frame, show = False, log_dir = video_log_dir, log_id = count)
                    file.write(f'{count}\t{cad_risk}\n')
                    print(cad_risk)
                count += 1
            else:
                break

        file.close()
        to_video(video_log_dir)
