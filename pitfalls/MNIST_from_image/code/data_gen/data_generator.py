import os
import cv2
import numpy as np

def data_generator(data_dir,train=True,batch_size = 4,num_classes=10):
    images = []
    labels = []
    for subdir, folders, files, in os.walk(data_dir):
        total = len(files)
        if train:
            parts = files[:int(total*0.8)]
        else:
            parts = files[int(total*0.8):]

        for file in parts:
            label = int(os.path.basename(subdir))
            image = cv2.imread(subdir+os.sep+file)[:,:,0:1]
            if (len(images)<batch_size):
                images.append(image)
                labels.append(label)
            else:
                images_arr = np.array(images)
                labels_arr = np.eye(num_classes)[labels]
                images = []
                labels = []
                yield(images_arr, labels_arr)


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    data_dir = os.path.join(project_dir,"data")
    tr_generator = data_generator(data_dir, num_classes = 10)
    img, label = next(tr_generator)
#    cv2.imwrite("./test.jpg",img)
    print(label)
    print(type(label))
