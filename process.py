# importing the libraries
import os
import shutil
import cv2
import numpy as np

IMG_SIZE = 224
RAW_PATH = 'raw'
DATASET_PATH = 'dataset'


def get_dim(width, height):
    if width > height:
        new_height = IMG_SIZE
        new_width = (new_height * width) / height
        return (new_height, int(new_width))
    else:
        new_width = IMG_SIZE
        new_height = (new_width * height) / width
        return (int(new_height), new_width)


if __name__ == '__main__':
    folders = os.listdir(RAW_PATH)

    if not os.path.exists(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    for folder in folders:
        print('Processing folder: {}'.format(folder))

        if os.path.exists(os.path.join(DATASET_PATH, folder)):
            shutil.rmtree(os.path.join(DATASET_PATH, folder))

        os.mkdir(os.path.join(DATASET_PATH, folder))

        imgs = os.listdir(os.path.join(RAW_PATH, folder))
        i = 0
        for img in imgs:
            input_path = os.path.join(RAW_PATH, folder, img)
            # print('Converting image: {}'.format(img))
            try:
                # Read the image
                data = cv2.imread(input_path)

                dim = get_dim(len(data), len(data[0]))

                data = cv2.resize(data, dim, interpolation=cv2.INTER_AREA)

                # Save as jpg
                output_path = os.path.join(
                    DATASET_PATH, folder, '{}.jpg'.format(i))

                long_side = max(dim)
                offset_long_side = int((long_side - IMG_SIZE) / 2)

                if dim[0] > dim[1]:
                    data = data[0:len(data),
                                0:IMG_SIZE]
                else:
                    data = data[0:IMG_SIZE, 0:len(data[0])]
                cv2.imwrite(output_path, data, [
                    int(cv2.IMWRITE_JPEG_QUALITY), 100])
                i += 1
            except Exception as e:
                print(e)
