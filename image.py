import os
import cv2
import numpy as np
from utils import get_file_name
import xml.etree.ElementTree as ET
from math import floor


def process_image(file_path, output_path, x, y, save_box_images, mode):
    (base_dir, file_name, ext) = get_file_name(file_path)
    image_path = os.path.join(base_dir, f"{file_name}.{ext}")
    xml = os.path.join(base_dir, f"{file_name}.xml")
    try:
        resize(
            image_path,
            xml,
            (x, y),
            output_path,
            mode,
            save_box_images=save_box_images
        )
    except Exception as e:
        print('[ERROR] error with {}\n file: {}'.format(image_path, e))
        print('--------------------------------------------------')


def draw_box(boxes, image, path):
    # Make a copy so we don't modify the source image if reused
    box_image = image.copy()
    for i in range(0, len(boxes)):
        cv2.rectangle(box_image, (boxes[i][2], boxes[i][3]), (boxes[i][4], boxes[i][5]), (255, 0, 0), 1)
    cv2.imwrite(path, box_image)


def resize(image_path,
           xml_path,
           newSize,
           output_path,
           mode,
           save_box_images=False,
           verbose=False
           ):

    (_, file_name, ext) = get_file_name(image_path)
    image = cv2.imread(image_path)
    delta_h = delta_w = top = bottom = left = right = 0
    mode = mode and mode.lower()
    # Standard resize mode
    if mode is None or mode == 'size':
        newSize = (int(newSize[0]), int(newSize[1]))
        scale_x = float(newSize[0]) / float(image.shape[1])
        scale_y = float(newSize[1]) / float(image.shape[0])
        image = cv2.resize(src=image, dsize=(newSize[0], newSize[1]))
    else:
        # Scaling by factor or percentage of the original image size
        if mode == 'scale' or mode == 'percentage':
            mul = 0.01 if mode == 'percentage' else 1.0
            newSize = (
                floor(float(image.shape[1]) * float(newSize[0]) * mul),
                floor(float(image.shape[0]) * float(newSize[1]) * mul))
            scale_x = newSize[0] / image.shape[1]
            scale_y = newSize[1] / image.shape[0]
            interp = cv2.INTER_LINEAR if (scale_x > 1.0 or scale_y > 1.0) else cv2.INTER_AREA
            image = cv2.resize(
                src=image,
                dsize=(0, 0), dst=None,
                fx=scale_x, fy=scale_y, interpolation=interp)
        # Target mode; choose the correct ratio to reach one of the x/y targets without oversize
        elif mode == 'target':
            ratio = float(int(newSize[0])) / float(image.shape[1])
            targetRatio = float(int(newSize[1])) / float(image.shape[0])
            ratio = targetRatio if targetRatio < ratio else ratio
            scale_x = scale_y = ratio
            interp = cv2.INTER_LINEAR if (scale_x > 1.0 or scale_y > 1.0) else cv2.INTER_AREA
            image = cv2.resize(
                src=image,
                dsize=(0, 0), dst=None,
                fx=scale_x, fy=scale_y, interpolation=interp)
        elif mode == 'pad':
            old_size = image.shape[:2]
            ratio = float(newSize[0]) / max(old_size)
            scale_x = scale_y = ratio
            new_size = tuple([int(x * ratio) for x in old_size])
            desired = int(newSize[0])
            image = cv2.resize(image, (new_size[1], new_size[0]))
            delta_w = desired - new_size[1]
            delta_h = desired - new_size[0]
            top, bottom = int(delta_h // 2), int(delta_h - (delta_h // 2))
            left, right = int(delta_w // 2), int(delta_w - (delta_w // 2))
            color = [0, 0, 0]
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        else:
            raise Exception(f"Invalid resize mode: {mode}")

    newBoxes = []
    has_xml = os.path.exists(xml_path)
    if has_xml:
        xmlRoot = ET.parse(xml_path).getroot()
        xmlRoot.find('filename').text = os.path.basename(image_path)
        size_node = xmlRoot.find('size')
        if mode == 'pad':
            size_node.find('width').text = str(desired)
            size_node.find('height').text = str(desired)
        else:
            size_node.find('width').text = str(newSize[0])
            size_node.find('height').text = str(newSize[1])

        for member in xmlRoot.findall('object'):
            bndbox = member.find('bndbox')

            xmin = bndbox.find('xmin')
            ymin = bndbox.find('ymin')
            xmax = bndbox.find('xmax')
            ymax = bndbox.find('ymax')

            xmin.text = str(int(np.round(float(xmin.text) * scale_x + left)))
            ymin.text = str(int(np.round(float(ymin.text) * scale_y + top)))
            xmax.text = str(int(np.round(float(xmax.text) * scale_x + left)))
            ymax.text = str(int(np.round(float(ymax.text) * scale_y + top)))

            newBoxes.append([
                1,
                0,
                int(float(xmin.text)),
                int(float(ymin.text)),
                int(float(xmax.text)),
                int(float(ymax.text))
            ])

        tree = ET.ElementTree(xmlRoot)
        tree.write(os.path.join(output_path, f"{file_name}.xml"))

    cv2.imwrite(os.path.join(output_path, '.'.join([file_name, ext])), image)

    if int(save_box_images) and has_xml:
        save_path = os.path.join(output_path, 'boxes_images', f"boxed_{file_name}.{ext}")
        draw_box(newBoxes, image, save_path)
