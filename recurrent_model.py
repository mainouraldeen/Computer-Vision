from tqdm import tqdm
import numpy as np
import cv2 as cv
import tensorflow.keras.utils as np_utils
import os
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, GRU, Dropout, TimeDistributed, Dense, Bidirectional, Conv3D, Flatten, MaxPooling3D, \
    ZeroPadding3D, BatchNormalization

"""**Global Variables**"""

y_labels = []
all_videos = []
max_no_frames = 0

training_folder = "Training_Set"
testing_folder = "Testing_Set"

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

            max_no_frames = max(max_no_frames, video_frames)

            one_video_frames = []
            one_video_frames = np.asarray(one_video_frames)
            leave_frame = 0
            while (True):
               # if leave_frame % 2 == 0:
                 ret, frame = cap.read()
                 if ret == False:
                     break
                 frame = cv.resize(frame, (80, 80))
                 frame = frame.reshape(80, 80, 3)
                 #frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                 one_video_frames = np.append(one_video_frames, frame)
                 one_video_frames = np.reshape(one_video_frames, (-1, frame.shape[0], frame.shape[1],3))
               # else:
                #   leave_frame += 1

                # all_videos.append(np.array([one_video_frames, label]))
            all_videos.append(one_video_frames)
            y_labels.append(label)
    print("max # of frames:", max_no_frames)
    y_labels = np_utils.to_categorical(y_labels)

    # all_videos: each element "has a diff size according to the length of each video" is a list of frames for each video
    np.save('all_videos.npy', all_videos)
    np.save('y_labels.npy', y_labels)
    np.save('max_no_frames.npy', max_no_frames)



def read_test_data():
  max_no_frames=0
  testing_videos=[]
  for video in os.listdir(testing_folder):

    video_path = os.path.join(testing_folder, video)
    cap = cv.VideoCapture(video_path)
    video_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    max_no_frames = max(max_no_frames, video_frames)

    one_video_frames = []
    one_video_frames = np.asarray(one_video_frames)

    leave_frame = 0
    while (True):
      ret, frame = cap.read()
      if ret == False:
        break
      # if leave_frame % 2 == 0:
          
      frame = cv.resize(frame, (80, 80))
      frame = frame.reshape(80, 80, 3)
      # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

      one_video_frames = np.append(one_video_frames, frame)
      one_video_frames = np.reshape(one_video_frames, (-1, frame.shape[0], frame.shape[1],3))
      # else:
      #     leave_frame += 1

    testing_videos.append(one_video_frames)
  return testing_videos

def main():
    global all_videos
    global y_labels
    global max_no_frames
    save_path = 'all_videos.npy'
    if (os.path.exists(save_path)):
        all_videos = np.load(save_path, allow_pickle=True)
        y_labels = np.load('y_labels.npy', allow_pickle=True)
        max_no_frames = np.load('max_no_frames.npy', allow_pickle=True)
        print("train data loaded")
    else:
        read_data()
        print("train data created")
   # read_data()
    padded_videos = []
    padded_videos = pad_sequences(all_videos, maxlen=max_no_frames, padding='pre')
    # padded_videos=pad_sequences(all_videos,padding='pre')# bt3rf lw7dha el max lenght <3
    print(len(padded_videos[1]))
    # print(all_videos.shape)

    ##################################################
    # model

    shape = (max_no_frames, 80, 80, 3)
    model = Sequential()

#   model.add(TimeDistributed(ZeroPadding3D(padding=(1, 2, 2), input_shape=shape)))
    model.add(Conv3D(32, kernel_size=(3,5,5),strides=(1, 2, 2), activation="relu", input_shape=shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

#   model.add(TimeDistributed(ZeroPadding3D(padding=(1, 2, 2))))
    model.add(Conv3D(64, kernel_size=(3, 5, 5), strides=(1, 1, 1), activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

#   model.add(TimeDistributed(ZeroPadding3D(padding=(1, 1, 1))))
    model.add(Conv3D(96, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(GRU(256, return_sequences=True), merge_mode='concat'))
    model.add(Bidirectional(GRU(256, return_sequences=True), merge_mode='concat'))

    model.add(Flatten())
    model.add((Dense(5, activation='softmax')))

   
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(padded_videos, y_labels, epochs=2, batch_size=max_no_frames, verbose=2)
    
    testing_videos= read_test_data()

    padded_test_videos = pad_sequences(testing_videos, maxlen=max_no_frames, padding='pre')


    prediction = model.predict(padded_test_videos)
    predicted_label=np.argmax(prediction,axis=1)

    print("----------predicted labels------------",predicted_label)

main()
