from torch.utils.data import DataLoader
import glob
import os
import xml.etree.ElementTree as ET
import numpy as np
import logging
import cv2

class YoloVocDataset:

    def __init__(self, root=None, transform=None, target_transform=None, is_test=False, keep_difficult=False):
        #Save parameters
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.keep_difficult = keep_difficult

        #Get xml, jpg file from folder
        self.label_files = glob.glob(root + "/*.xml")
        self.image_files = [f[0:-4] + ".jpg" for f in self.label_files]
        self.class_names = ('BACKGROUND', 'cone', 'cube', 'sphere')
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}
        print('self.class_dict = ', self.class_dict)
        self.check_valid_files()
        # print(self.label_files[10])
        # print(self.image_files[10])

    def check_valid_files(self):
        count = 0
        for f in self.label_files:
            if os.path.exists(f) == False:
                print(f)
                count += 1
        print(f'missing label_files = {count} / {len(self.label_files)} total')

        for f in self.image_files:
            if os.path.exists(f) == False:
                print(f)
                count += 1
        print(f'missing image_files = {count} / {len(self.image_files)} total')


    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, index):
        boxes, labels, is_difficult = self.get_annotation(index)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self.read_image(index)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        #print(f'__getitem__[{index}]: {image.shape}, {boxes.shape}, {labels.shape}')
        return image, boxes, labels
        #return image

    def read_image(self, index):
        path_file = self.image_files[index]
        image = cv2.imread(str(path_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #if self.transform:
        #    image, _ = self.transform(image)
        # print(f'read_image[{index}] = {image.shape}')
        return image

    def get_annotation(self, index):
        path_file = self.label_files[index]
        objects = ET.parse(path_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find('bndbox')

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

import multiprocessing
import torch
def custom_collate(batch):
    '''
    images, boxes, labels = zip(*batch)
   
    images = [torch.tensor(i) for i in images]
    boxes = [torch.tensor(i) for i in boxes]
    labels = [torch.tensor(i) for i in labels]
    return  torch.stack(list(images), dim=0), torch.cat(boxes, dim=0), torch.cat(labels, dim=0) 
    '''
    return images, boxes, lables

if __name__ == '__main__':
    multiprocessing.freeze_support()
    train_dataset = YoloVocDataset('data/3D_shapes_yolo_voc/test')
    train_loader = DataLoader(train_dataset, batch_size=4, num_workers=1, shuffle=False, collate_fn=custom_collate)

    for i, data in enumerate(train_loader):
        images, boxes, labels = data
        print(f'Main {i+1} images = ', images.shape)
        print(f'Main {i+1} boxes = ', boxes.shape)
        print(f'Main {i+1} labels = ', labels.shape)
        #images = data
        #print(f'Main {i+1} images = ', images.shape)
