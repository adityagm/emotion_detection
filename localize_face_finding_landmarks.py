# localizing the faces in the frame. here only one face has to be localized.

import cv2
import numpy as np
import dlib
import pandas
from neural_net_svm import NeuralNetSvm
from sklearn.utils import shuffle
from pathlib import Path
import os
import csv

# to store all the frames with the bounding box over the localized faces
# use HoG to detect faces
find_face = dlib.get_frontal_face_detector()

# display the image with the bounding box
win_show_image = dlib.image_window()

# load pre-trained models from dlib to detect the facial landmarks
shape_predictor = 'E:\Wenger\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat'

# create the detector using the above pretrained model
face_pose_detector = dlib.shape_predictor(shape_predictor)

# text file opened to save the landmarks
fp = open("landmarks_curious.txt","wb")

def localize_face_in_frame(path_in, landmarks = []): #to store landmarks
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # landmarks index

    returned_landmarks = []
    for file in Path(path_in).glob('*.jpg'):
        grey = cv2.imread('{}'.format(file),0)
        image = clahe.apply(grey)
        returned_landmarks = find_landmarks_of_faces_and_return_landmarks(image)
##        print(file)
        if returned_landmarks:
            landmarks.append(find_landmarks_of_faces_and_return_landmarks(image))
            # no need to insert the name field into the landmarks dataset since it will be
            # passed to the svm as a separate list of labels
##            landmarks.insert(135, name)
##            index+=1
##            
##            print("length of landmarks", len(landmarks))
##            print(landmarks)
    # misbehaving
##    pickle.dump(landmarks,fp)
    return landmarks

def find_landmarks_of_faces_and_return_landmarks(read_image):
##      resize the image to 50%
    scale_percent = 30 # percent of original size
    width = int(read_image.shape[1] * scale_percent / 100)
    height = int(read_image.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image, stores as float64 with range 0-1
    resized = cv2.resize(read_image, dim, interpolation = cv2.INTER_AREA)

    detected_faces = find_face(resized,1)
##    print("I found {} faces in the file {}".format(len(detected_faces), path_in))
    win_show_image.clear_overlay()
    win_show_image.set_image(resized)
    face_landmarks = []
    # Loop through each face we found in the image
    for i, face_rect in enumerate(detected_faces):

        # Detected faces are returned as an object with the coordinates 
        # of the top, left, right and bottom edges
##        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

        # Draw a box around each face we found
        win_show_image.add_overlay(face_rect)
##        print("face_rect",face_rect)
    
        pose_landmarks = face_pose_detector(resized,face_rect)
        xlist = []
    
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(pose_landmarks.part(i).x))
            ylist.append(float(pose_landmarks.part(i).y))

        # obtain the mean of the two data point lists x and y

        x_mean = np.mean(xlist)
        y_mean = np.mean(ylist)

        # calculate the variance of the points

        x_central = [(x-x_mean) for x in xlist]
        y_central = [(y-y_mean) for y in ylist]

        for z, w, x, y in zip(x_central, y_central, xlist, ylist): #Store all landmarks in one list in the format x1,y1,x2,y2,etc.
            face_landmarks.append(x)
            face_landmarks.append(y)
            meannp = np.asarray((y_mean, x_mean))
            coornp = np.asarray((y,x))
            dist = np.linalg.norm(coornp - meannp)
            face_landmarks.append(dist)
            face_landmarks.append((np.arctan2(w,z)*360)/(2*np.pi))
            
        win_show_image.add_overlay(pose_landmarks)
##        print(len(face_landmarks))
        return(face_landmarks)
            

def train(rootdir):
    
    train_landmarks = []
    train_labels = []
    for subdirs, dirs, files in os.walk(rootdir):
        name = os.path.basename(subdirs)
        
        e_landmark = localize_face_in_frame(subdirs)
        for l in e_landmark:
            train_landmarks.append(l)
            train_labels.append(name)
    
##        print("no of train datasets:",len(train_landmarks),"and corresponding labels: %d", len(train_labels))
    train_landmarks, train_labels = shuffle(train_landmarks, train_labels, random_state = 0)
    #print("train_labels:", train_labels)
    if((os.path.isfile('train_landmark_csv_try9.csv'))!=1):
        with open('train_landmark_csv_try9.csv','w') as f_csv:
            writer = csv.writer(f_csv, dialect='excel')

            for landmark, label in zip(train_landmarks,train_labels):
                # following syntax writes the landmarks into one column as a string and the labels into another
                # which is undesirable, since i want to read the landmarks as individual variables
##                writer.writerow([','.join(map(str,landmark)),label])

               # better the following way
               writer.writerow([landmark,label])

                

##    with open('train_landmark_csv.csv','r') as f_csv:
##	spamreader = csv.reader(f_csv, delimiter=',')
##	for row in spamreader:
##		l.append(row)
    
    train_landmarks_np = np.array(train_landmarks)
    train_labels_np = np.array(train_labels)
    
    nn_svm.fit(train_landmarks_np, train_labels_np)
    return

def validate(rootdir):

    validation_landmarks = [None]
    i = 0
    predictions = []
    test_unit = []
    with open('results_try6.csv','w') as r_csv:
        writer = csv.writer(r_csv, dialect='excel')
        for file in Path(rootdir).glob('*.jpg'):
            image = cv2.imread('{}'.format(file),0)
            
    ##        validation_landmarks[0] = find_landmarks_of_faces_and_return_landmarks(image)
            landmarks = find_landmarks_of_faces_and_return_landmarks(image)
            if(landmarks):
                i += 1
                validation_landmarks[0] = landmarks
                validation_landmarks_np = np.array(validation_landmarks)
                prediction = nn_svm.predict(validation_landmarks_np)
                predictions.append(prediction)
                test_unit.append('{}'.format(file))
                print("prediction",prediction)
                writer.writerow([test_unit[len(test_unit)-1], predictions[len(predictions)-1]])

    return
    
        

        
            # following syntax writes the landmarks into one column as a string and the labels into another
            # which is undesirable, since i want to read the landmarks as individual variables
##                writer.writerow([','.join(map(str,landmark)),label])

           # better the following way
          
    
def main():
    
    train_path = 'E:\Wenger\emotions'
    validate_path = 'E:\Wenger\data_all_vids0_gray'
    global nn_svm
    nn_svm = NeuralNetSvm()
    
    train(train_path)
    validate(validate_path)

    

    

if __name__ == '__main__':
    main()
