from tqdm import tqdm
import pandas as pd
from scipy import stats as s
import numpy as np
from statistics import mode#,multimode
import cv2 as cv
import tensorflow.keras.utils as np_utils
import os
from sklearn.utils import shuffle
#from random import shuffle
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, GRU, Dropout, TimeDistributed, Dense, Bidirectional, Conv3D, Flatten, MaxPooling3D, \
    ZeroPadding3D, BatchNormalization

"""**Global Variables**"""

y_labels = []
all_videos = []
max_no_frames = 0

training_folder = "Training_set"
testing_folder = "Testing_set"
submitFile = pd.read_csv('submit2.csv')

"""**Get max number of frames and saving all videos without duplication in (total_frames)**"""


# read data and labeling, return max number in all frames
def read_data():
    global all_videos
    global max_no_frames
    global y_labels

    for folder in tqdm(os.listdir(training_folder)):  # folder: Basketball, Diving, Jumping, Tennis, and Walking
        folder_path = os.path.join(training_folder, folder)
        print("-->In folder:", folder)
        # check_counter = 0
        label = -1
        # naming creiteria:
        if (folder == "Diving"):
            label = 0
        elif (folder == "Jumping"):
            label = 1
        elif (folder == "Basketball"):
            label = 2
        elif (folder == "Tennis"):
            label = 3
        elif (folder == "Walking"):
            label = 4

        for video in os.listdir(folder_path):

            video_path = os.path.join(folder_path, video)
            cap = cv.VideoCapture(video_path)
            video_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

            max_no_frames = max(max_no_frames, int(video_frames/2))

            one_video_frames = []
            one_video_frames = np.asarray(one_video_frames)
            leave_frame = 0
            sub_videos = 0
            while (True):
                ret, frame = cap.read()
                if ret == False:
                   cap.release()
                   break

                frame = cv.resize(frame, (200, 200))
                frame = frame.reshape(200, 200, 3)
               # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                one_video_frames = np.append(one_video_frames, frame)
                one_video_frames = np.reshape(one_video_frames, (-1, frame.shape[0], frame.shape[1],3))

                if (leave_frame+2 <video_frames):
                   leave_frame += 2
                   cap.set(1,leave_frame)
                # all_videos.append(np.array([one_video_frames, label]))
                
                if (sub_videos==30):
                   all_videos.append(one_video_frames)
                   y_labels.append(label)
                   one_video_frames=[]
                   sub_videos=0
                sub_videos+=1              
           # all_videos.append(one_video_frames)
           # print("video lenght read data", len(one_video_frames))
           # y_labels.append(label)
    print("max # of frames:", max_no_frames)
    y_labels = np_utils.to_categorical(y_labels)
    print("Lenn ylabels", len(y_labels))
    print("LENNN all_videos",len(all_videos))
    # all_videos: each element "has a diff size according to the length of each video" is a list of frames for each video
    shuffle(all_videos,y_labels) #lazm lama a shuffle arbot bl label    
    np.save('all_videosP1.npy', all_videos[ :600])
    np.save('all_videosP2.npy', all_videos[600:])
    np.save('y_labels.npy', y_labels)
    np.save('max_no_frames.npy', max_no_frames)



def read_test_data():
  max_no_frames=0
  testing_videos=[]
  videos_names=[]
  for video in tqdm(os.listdir(testing_folder)):
    videos_names.append(video)
    video_path = os.path.join(testing_folder, video)
    cap = cv.VideoCapture(video_path)
    video_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    max_no_frames = max(max_no_frames, video_frames)

    one_video_frames = []
    one_video_frames = np.asarray(one_video_frames)
    sub_videos=0
    test_video=[]
    leave_frame = 0
    while (True):
      ret, frame = cap.read()
      if ret == False:
        break
      # if leave_frame % 2 == 0:
          
      frame = cv.resize(frame, (200, 200))
      frame = frame.reshape(200, 200, 3)
     # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

      one_video_frames = np.append(one_video_frames, frame)
      one_video_frames = np.reshape(one_video_frames, (-1, frame.shape[0], frame.shape[1],3))
      if (sub_videos==30):
         test_video.append(one_video_frames)
         one_video_frames=[]
         sub_videos=0
      sub_videos+=1
    testing_videos.append(test_video)
  np.save('testing_videos.npy', testing_videos)
  np.save('videos_names.npy', videos_names)
  return testing_videos, videos_names

def main():
    global all_videos
    global y_labels
    global max_no_frames
    save_path = 'all_videos.npy'
    
   # if (os.path.exists(save_path)):
    #    all_videos = np.load(save_path, allow_pickle=True)
     #   y_labels = np.load('y_labels.npy', allow_pickle=True)
      #  max_no_frames = np.load('max_no_frames.npy', allow_pickle=True)
      # # read_data()        
       # print("train data loaded")
    #else:
     #   read_data()
      #  print("train data created")
    read_data()
    padded_videos = []
    print("max no frames main", max_no_frames)
    padded_videos = pad_sequences(all_videos, maxlen=30, padding='pre')
    # padded_videos=pad_sequences(all_videos,padding='pre')# bt3rf lw7dha el max lenght <3

    ##################################################
    # training model

#    shape = (max_no_frames, 80, 80,1)
    shape = (30, 200, 200,3)
 
  # print("-------",padded_videos[0].shape)
    model = Sequential()

#    model.add(TimeDistributed(ZeroPadding3D(padding=(1, 2, 2), input_shape=shape)))
    model.add(Conv3D(32, kernel_size=(3,5,5),strides=(1, 2, 2), activation="relu", input_shape=shape))
    model.add(BatchNormalization())
#    model.add(Dropout(0.5))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

#    model.add(TimeDistributed(ZeroPadding3D(padding=(1, 2, 2))))
    model.add(Conv3D(64, kernel_size=(3, 5, 5), strides=(1, 1, 1), activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

#    model.add(TimeDistributed(ZeroPadding3D(padding=(1, 1, 1))))
    model.add(Conv3D(96, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu"))
    model.add(BatchNormalization())
 #   model.add(Dropout(0.5))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    #model.add(TimeDistributed(Flatten()))

    #model.add(Bidirectional(GRU(32, return_sequences=True), merge_mode='concat'))
    #model.add(Bidirectional(GRU(32, return_sequences=True), merge_mode='concat'))

    model.add(Flatten())
    model.add((Dense(124, activation='relu')))
    model.add((Dense(5, activation='softmax')))

   
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    #model.fit(padded_videos[:20], y_labels[:20], epochs=2, batch_size=max_no_frames, verbose=2)
    model.fit(padded_videos, y_labels, epochs=15, batch_size=30, verbose=2)
    model.save("recurrent_model.h5")
    print("model is saved") 
    #testing
    videos_names=[]
    testing_videos=[]
   # if (os.path.exists("testing_videos.npy")):
    #    testing_videos = np.load("testing_videos.npy", allow_pickle=True)
     #   videos_names = np.load("videos_names.npy", allow_pickle=True)
      ##  testing_videos, videos_names= read_test_data()

    #else:
     #  testing_videos, videos_names= read_test_data()
    testing_videos, videos_names= read_test_data()

   # padded_test_videos = pad_sequences(testing_videos, maxlen=max_no_frames, padding='pre')

    predicted_labels=[]
    print("LEENN", len(testing_videos))
    print("LEENN 0", len(testing_videos[0]))
    print("LEENN 1", len(testing_videos[1]))

    for i in range(len(testing_videos)):
        padded_test = pad_sequences(testing_videos[i], maxlen=30, padding='pre')
        prediction=model.predict(padded_test)
        prediction=np.argmax(prediction,axis=1)
       
       # majority_voting=multimode(prediction)
        majority_voting=s.mode(prediction)[0]
        predicted_labels.append(majority_voting)
   # prediction = model.predict(padded_test_videos)
   # predicted_labels=np.argmax(prediction,axis=1)

    submitFile['Label'] = predicted_labels
    submitFile['Video'] = videos_names
    submitFile.to_csv('submitFile2.csv', index=False)
    print("Submit File Saved Successfully")


main()
