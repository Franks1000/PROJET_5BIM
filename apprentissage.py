import cv2 
import os 
import numpy as np 
import pickle 

# py -3 -m pip install numpy pickle 

image_dir = "./images/" # repository
current_id = 0 
label_ids = {} 
x_train = [] 
y_labels = [] 

for root, dirs, files in os.walk(image_dir):

   # print("Directory path: %s"%root)
   # print("Directory Names: %s"%dirs)
   # print("Files Names: %s"%files) 

    # if files 
    if len(files):
        label = root.split("/")[-1]
        for file in files:
            if file.endswith("png"):
                path = os.path.join(root, file)
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
                image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (200, 200))
                x_train.append(image)
                y_labels.append(id_)

with open("labels.pickle", "wb") as f:
    a = pickle.dump(label_ids, f)
x_train = [cv2.resize(img, (200, 200)) for img in x_train]
x_train = np.array(x_train)
y_labels = np.array(y_labels)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(x_train, y_labels)
recognizer.save("trainner.yml")
print(x_train, y_labels)
recognizer.train(x_train, y_labels)
recognizer.save("trainner.yml")





