import sys
sys.path.append('..')
import third_party.yolo_world.yoloworld_detector as yoloworld_detector

runner = yoloworld_detector.yoloworld_preload()

import os
import matplotlib.pyplot as plt
root = '/home/zhouj0d/Science/PID26.EWS/EWS/dataset/CAD/cad'
for patient_id in os.listdir(root):
    if os.path.isdir('{}/{}'.format(root, patient_id)):
        patient_root = '{}/{}'.format(root, patient_id)
        print('Processing {}'.format(patient_root))
        for image in ['left', 'right', 'front', 'top']:
            if True:
            #if not os.path.isfile(f'{patient_root}/{image}.CroppedBox.png'):
                if os.path.isfile(f'{patient_root}/{image}.jpg'):
                    img_path = '{}/{}.jpg'.format(patient_root, image)
                    output_image, detections, texts, labels, crops = yoloworld_detector.yoloworld_predict_single_image(runner = runner, img_path = img_path, input_text='head')
                    crop = crops[0]
                    crop.save(f'{patient_root}/{image}.CroppedBox.png')
                    #plt.imshow(crop)
                    #plt.show()
                    #plt.close()
                elif os.path.isfile(f'{patient_root}/{image}.JPG'):
                    img_path = '{}/{}.JPG'.format(patient_root, image)
                    output_image, detections, texts, labels, crops = yoloworld_detector.yoloworld_predict_single_image(runner = runner, img_path = img_path, input_text='head')
                    crop = crops[0]
                    crop.save(f'{patient_root}/{image}.CroppedBox.png')
                    #plt.imshow(crop)
                    #plt.show()
                    #plt.close()
            else:
                print('** Exist @{}/{}!'.format(patient_root, image))
                
root = '/home/zhouj0d/Science/PID26.EWS/EWS/dataset/CAD/normal'
for patient_id in os.listdir(root):
    if os.path.isdir('{}/{}'.format(root, patient_id)):
        patient_root = '{}/{}'.format(root, patient_id)
        print('Processing {}'.format(patient_root))
        for image in ['left', 'right', 'front', 'top']:
            if True:
            #if not os.path.isfile(f'{patient_root}/{image}.CroppedBox.png'):
                if os.path.isfile(f'{patient_root}/{image}.jpg'):
                    img_path = '{}/{}.jpg'.format(patient_root, image)
                    output_image, detections, texts, labels, crops = yoloworld_detector.yoloworld_predict_single_image(runner = runner, img_path = img_path, input_text='head')
                    crop = crops[0]
                    crop.save(f'{patient_root}/{image}.CroppedBox.png')
                    #plt.imshow(crop)
                    #plt.show()
                    #plt.close()
                elif os.path.isfile(f'{patient_root}/{image}.JPG'):
                    img_path = '{}/{}.JPG'.format(patient_root, image)
                    output_image, detections, texts, labels, crops = yoloworld_detector.yoloworld_predict_single_image(runner = runner, img_path = img_path, input_text='head')
                    crop = crops[0]
                    crop.save(f'{patient_root}/{image}.CroppedBox.png')
                    #plt.imshow(crop)
                    #plt.show()
                    #plt.close()
            else:
                print('** Exist @{}/{}!'.format(patient_root, image))