import cv2
import numpy as np
import os

haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
datasets = 'datasets'

print('Training....')

(images, labels, names, id) = ([], [], {}, 0)
target_size = (130, 100)  # Ensure all images are resized to this size

# Debug: Print the structure of the datasets directory
print("Walking through the datasets directory...")

for (subdirs, dirs, files) in os.walk(datasets):
    print(f"Subdirectories: {dirs}")
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        print(f"Processing directory: {subjectpath}")
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            print(f"Reading image: {path}")
            img = cv2.imread(path, 0)
            if img is not None:
                img = cv2.resize(img, target_size)  # Resize image to target size
                images.append(img)
                labels.append(int(label))
            else:
                print(f"Failed to read image: {path}")
        id += 1

# Check if images and labels have been populated
if len(images) == 0 or len(labels) == 0:
    raise ValueError("No images found in the dataset directory.")

(images, labels) = [np.array(lis) for lis in [images, labels]]
print(f"Images: {len(images)}, Labels: {len(labels)}")

model = cv2.face.LBPHFaceRecognizer_create()
# model = cv2.face.FisherFaceRecognizer_create()

model.train(images, labels)

webcam = cv2.VideoCapture(0)
cnt = 0

while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, target_size)

        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 2)
        if prediction[1] < 800:
            cv2.putText(im, '%s - %.0f' % (names[prediction[0]], prediction[1]), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
            print(names[prediction[0]])
            cnt = 0
        else:
            cnt += 1
            cv2.putText(im, 'Unknown', (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
            if cnt > 100:
                print("Unknown Person")
                cv2.imwrite("Unknown.jpg", im)
                cnt = 0
    cv2.imshow('FaceRecognition', im)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()
