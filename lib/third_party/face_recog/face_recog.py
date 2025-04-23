import face_recognition
from torchvision import datasets
import os
from torch.utils.data import DataLoader
import torch
import cv2
import numpy as np
import supervision as sv
from PIL import Image
BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness = 5)
LABEL_ANNOTATOR = sv.LabelAnnotator(text_color=sv.Color.BLACK)

class ImageFolderWithPaths(datasets.ImageFolder):

    def __getitem__(self, index):
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        return (img, label, path)

def extend_bounding_box(x1, y1, x2, y2, extend_ratio, img_width, img_height):
    # 计算原始 bounding box 的宽度和高度
    box_width = x2 - x1
    box_height = y2 - y1
    
    # 计算扩展后的宽度和高度
    extended_width = box_width * extend_ratio
    extended_height = box_height * extend_ratio
    
    # 计算扩展后的坐标
    new_x1 = int(max(0, x1 - (extended_width - box_width) / 2))
    new_y1 = int(max(0, y1 - (extended_height - box_height) / 2))
    new_x2 = int(min(img_width, x2 + (extended_width - box_width) / 2))
    new_y2 = int(min(img_height, y2 + (extended_height - box_height) / 2))
    
    return new_x1, new_y1, new_x2, new_y2

class FaceDetector:
    def __init__(self, recognition = False, user_face_reference_img_path = None, model = None, tolerance = None):
        self.face_embedding = None
        self.user_face_reference_img_path = user_face_reference_img_path
        self.number_of_times_to_upsample = 1
        self.recognition = recognition
        self.model = model
        self.tolerance = tolerance
        if self.recognition:
            self.load_embedding()

    def process_one_cv2_frame(self, frame, x1 = 0, y1 = 0, crop = True, extend_ratio = None):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        img_height, img_width = frame.shape[:2]
        # _y1, _x2, _y2, _x1
        face_locations = face_recognition.face_locations(frame_rgb,
                                                         number_of_times_to_upsample=self.number_of_times_to_upsample,
                                                         model=self.model)

        if self.recognition:
            face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
            face_names = []
            known_face_encodings, known_face_names = self.face_embedding_data[0], self.face_embedding_data[1]
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance = self.tolerance)
                name = "N/A"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)

        new_face_locations = []
        if len(face_locations) > 0:
            for _ in face_locations:
                (_y1, _x2, _y2, _x1) = _
                if extend_ratio:
                    new_x1, new_y1, new_x2, new_y2 = extend_bounding_box(_x1, _y1, _x2, _y2, extend_ratio, img_width, img_height)
                    new_face_locations.append((new_x1, new_y1, new_x2, new_y2))
                else:
                    new_face_locations.append((_x1, _y1, _x2, _y2))
        
        if self.recognition:
            face_names = [f'face_name@{_}' for _ in face_names]
        else:
            face_names = ['face_name@Off' for _ in range(len(face_locations))]
        
        detections = None
        labels = face_names
        texts = None
        crops = []
        
        if len(new_face_locations) > 0:
            class_ids = []
            for _ in range(len(new_face_locations)):
                class_ids.append(0)
            detections = sv.Detections(xyxy = np.array(new_face_locations), class_id = np.array(class_ids))
            if crop:
                for xyxy in detections.xyxy:
                    cropped_image = image.crop(xyxy)
                    crops.append(cropped_image)
                
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
            #print(detections,labels)
            image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
                
        return image, detections, texts, labels, crops

    def load_embedding(self):
        if os.path.isfile('{}/face_embedding_data.pt'.format(self.user_face_reference_img_path)) == False:
            print('>> generate face embedding')
            dataset = ImageFolderWithPaths(self.user_face_reference_img_path)  # photos folder path
            idx_to_class = {i: c for c, i in
                            dataset.class_to_idx.items()}  # accessing names of peoples from folder names

            def collate_fn(x):
                return x[0]

            loader = DataLoader(dataset, collate_fn=collate_fn)

            known_face_encodings = []  # list of names corrospoing to cropped photos
            known_face_names = []  # list of embeding matrix after conversion from cropped faces to embedding matrix using resnet

            for img, idx, path in loader:
                print(path)
                image = face_recognition.load_image_file(path)
                encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append(encoding)
                known_face_names.append(idx_to_class[idx])

            data = [known_face_encodings, known_face_names]
            torch.save(data, '{}/face_embedding_data.pt'.format(self.user_face_reference_img_path))  # saving data.pt file
            load_data = data
        else:
            print('>> face embedding exist, loading')
            load_data = torch.load('{}/face_embedding_data.pt'.format(self.user_face_reference_img_path))
        self.face_embedding_data = load_data


if __name__ == '__main__':
    import dlib
    print(dlib.DLIB_USE_CUDA) # test CUDA support
    print(dlib.cuda.get_num_devices())
    
    face_detector = FaceDetector(recognition=False, user_face_reference_img_path='../../user_profile')
    video_path = '/home/zhouj0d/Science/PID26.EWS/EWS/DigitalShadow/test_data/crowd.mp4'

    try:
        # 打开摄像头
        cap = cv2.VideoCapture(int(float(video_path)))
        print(f'open camera {video_path}')
    except:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        print(f'open {video_path}')

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            output_image, detections, texts, labels, crops = face_detector.process_one_cv2_frame(frame = frame)
            print(detections)