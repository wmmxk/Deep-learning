import os
import cv2
import numpy as np

def data_generator_correct(data_dir,train=True,batch_size = 4,num_classes = 10):
    file_paths = []
    for subdir, folders, files, in os.walk(data_dir):
        for file in files:
            file_paths.append(subdir+os.sep+file)
    random_index = np.arange(len(file_paths))
    np.random.shuffle(random_index)
    file_paths = [file_paths[i] for i in random_index]
    total = len(file_paths)

    if train:
        parts = file_paths[:int(total*0.8)]
    else:
        parts = file_paths[int(total*0.8):]
    images = []
    labels = []
    while True:
        for start in range(0,len(parts),batch_size):
            end = min(start+batch_size, len(parts))
            for path in parts[start:end]:
                image = cv2.imread(path)[:,:,0:1]/250.0
                label = int(path.split("/")[-2])
                images.append(image)
                labels.append(label)
            images_arr = np.array(images)
            labels_arr = np.eye(num_classes)[labels]
            images = []
            labels = []
            yield(images_arr, labels_arr)


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    data_dir = os.path.join(project_dir,"data")
    tr_generator = data_generator_correct(data_dir)
    img,label = next(tr_generator)
#    cv2.imwrite("./test.jpg",img)
    print(img.shape,label)
