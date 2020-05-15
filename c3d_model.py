from tqdm import tqdm
import pandas as pd
from scipy import stats
import numpy as np
from keras.models import load_model
from statistics import mode  # ,multimode
import cv2 as cv
import keras.utils as np_utils
import os
from sklearn.utils import shuffle
# from random import shuffle
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, GRU, Dropout, TimeDistributed, Dense, Bidirectional, Conv3D, Flatten, MaxPooling3D, \
    ZeroPadding3D, BatchNormalization,Convolution3D

"""**Global Variables**"""

y_labels = []
all_videos = []
max_no_frames = 0

training_folder = "Training_set"
testing_folder = "Testing_set"
labeled_test_folder = "labeled_test_set"
# labeled_test_folder = "Training_Set"

submitFile = pd.read_csv('submit.csv')

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

            max_no_frames = max(max_no_frames, int(video_frames / 2))

            one_video_frames = []
            one_video_frames = np.asarray(one_video_frames)
            leave_frame = 0
            sub_videos = 0
            while (True):
                ret, frame = cap.read()
                if ret == False:
                    cap.release()
                    break

                frame = cv.resize(frame, (112, 112))
                frame = frame.reshape(112,112, 3)
                # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                one_video_frames = np.append(one_video_frames, frame)
                one_video_frames = np.reshape(one_video_frames, (-1, frame.shape[0], frame.shape[1], 3))

                if (leave_frame + 2 < video_frames):
                    leave_frame += 2
                    cap.set(1, leave_frame)
                # all_videos.append(np.array([one_video_frames, label]))

                if (sub_videos == 16):
                    all_videos.append(one_video_frames)
                    y_labels.append(label)
                    one_video_frames = []
                    sub_videos = 0
                sub_videos += 1
                # all_videos.append(one_video_frames)
        # print("video lenght read data", len(one_video_frames))
        # y_labels.append(label)
    print("max # of frames:", max_no_frames)
    y_labels = np_utils.to_categorical(y_labels)
    print("Lenn ylabels", len(y_labels))
    print("LENNN all_videos", len(all_videos))
    # all_videos: each element "has a diff size according to the length of each video" is a list of frames for each video
    shuffle(all_videos, y_labels)  # lazm lama a shuffle arbot bl label
    #np.save('all_videosP1.npy', all_videos[:600])
    #np.save('all_videosP2.npy', all_videos[600:])
    np.save('y_labels.npy', y_labels)
    np.save('max_no_frames.npy', max_no_frames)


def read_labeled_test_data():
    videos_names = []
    test_labels = []
    test_videos = []

    for folder in os.listdir(labeled_test_folder):  # folder: Basketball, Diving, Jumping, Tennis, and Walking
        folder_path = os.path.join(labeled_test_folder, folder)
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

        for video in tqdm(os.listdir(folder_path)):
            videos_names.append(video)
            video_path = os.path.join(folder_path, video)
            cap = cv.VideoCapture(video_path)
            video_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

            one_video_frames = []
            one_video_frames = np.asarray(one_video_frames)
            leave_frame = 0
            sub_videos = 0
            test_video = []

            while (True):
                ret, frame = cap.read()
                if ret == False:
                    cap.release()
                    break

                frame = cv.resize(frame, (112, 112))
                frame = frame.reshape(112, 112, 3)
                # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                one_video_frames = np.append(one_video_frames, frame)
                one_video_frames = np.reshape(one_video_frames, (-1, frame.shape[0], frame.shape[1], 3))

                if (sub_videos ==16):
                    test_video.append(one_video_frames)
                    # test_labels.append(label)
                    one_video_frames = []
                    sub_videos = 0
                sub_videos += 1
            test_videos.append(test_video)
            test_labels.append(label)

    # test_labels2=test_labels
    # test_labels = np_utils.to_categorical(test_labels)

    # np.save('test_videos.npy', test_videos)
    # np.save('test_labels.npy', test_labels)
    # np.save('videos_names.npy', videos_names)
    return test_videos, test_labels, videos_names


def read_test_data():
    max_no_frames = 0
    testing_videos = []
    videos_names = []
    for video in tqdm(os.listdir(testing_folder)):
        videos_names.append(video)
        video_path = os.path.join(testing_folder, video)
        cap = cv.VideoCapture(video_path)
        video_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

        max_no_frames = max(max_no_frames, video_frames)

        one_video_frames = []
        one_video_frames = np.asarray(one_video_frames)
        sub_videos = 0
        test_video = []
        leave_frame = 0
        while (True):
            ret, frame = cap.read()
            if ret == False:
                break
            # if leave_frame % 2 == 0:

            frame = cv.resize(frame, (112, 112))
            frame = frame.reshape(112, 112, 3)
            # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            one_video_frames = np.append(one_video_frames, frame)
            one_video_frames = np.reshape(one_video_frames, (-1, frame.shape[0], frame.shape[1], 3))
            if (sub_videos ==16):
                test_video.append(one_video_frames)
                one_video_frames = []
                sub_videos = 0
            sub_videos += 1
        testing_videos.append(test_video)
    #np.save('testing_videos.npy', testing_videos)
    #np.save('videos_names.npy', videos_names)
    return testing_videos, videos_names

def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False
def main():
    global all_videos
    global y_labels
    global max_no_frames
    # save_path = 'all_videos.npy'
    save_path = 'all_videosP1.npy'

    if (os.path.exists(save_path)):
        all_videos = np.load(save_path, allow_pickle=True)
        x = np.load("all_videosP2.npy", allow_pickle=True)
        all_videos = np.append(all_videos, x)
        y_labels = np.load('y_labels.npy', allow_pickle=True)
        # max_no_frames = np.load('max_no_frames.npy', allow_pickle=True)
        print("train data loaded")
    else:
        read_data()
        print("train data created")

    # if (os.path.exists(save_path)):
    #     all_videos = np.load(save_path, allow_pickle=True)
    #     y_labels = np.load('y_labels.npy', allow_pickle=True)
    #     max_no_frames = np.load('max_no_frames.npy', allow_pickle=True)
    #     print("train data loaded")
    # else:
    #     read_data()
    #     print("train data created")
    # read_data()

    padded_videos = pad_sequences(all_videos, maxlen=16, padding='pre')
    # padded_videos=pad_sequences(all_videos,padding='pre')# bt3rf lw7dha el max lenght <3


    # model = load_model('recurrent_model.h5')

    # training model
    shape = (16, 112, 112, 3)
    model = Sequential()
    # 1st layer group
    model.add(Conv3D(64, (3, 3, 3), activation="relu", name="conv1",
                     input_shape=shape,
                     strides=(1, 1, 1), padding="same"))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), name="pool1", padding="valid"))

    # 2nd layer group
    model.add(Conv3D(128, (3, 3, 3), activation="relu", name="conv2",
                     strides=(1, 1, 1), padding="same"))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool2", padding="valid"))

    # 3rd layer group
    model.add(Conv3D(256, (3, 3, 3), activation="relu", name="conv3a",
                     strides=(1, 1, 1), padding="same"))
    model.add(Conv3D(256, (3, 3, 3), activation="relu", name="conv3b",
                     strides=(1, 1, 1), padding="same"))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool3", padding="valid"))

    # 4th layer group
    model.add(Conv3D(512, (3, 3, 3), activation="relu", name="conv4a",
                     strides=(1, 1, 1), padding="same"))
    model.add(Conv3D(512, (3, 3, 3), activation="relu", name="conv4b",
                     strides=(1, 1, 1), padding="same"))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool4", padding="valid"))

    # 5th layer group
    model.add(Conv3D(512, (3, 3, 3), activation="relu", name="conv5a",
                     strides=(1, 1, 1), padding="same"))
    model.add(Conv3D(512, (3, 3, 3), activation="relu", name="conv5b",
                     strides=(1, 1, 1), padding="same"))
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool5", padding="valid"))
    model.add(Flatten())

    # FC layers group
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(.5))
    model.add(Dense(4096, activation='relu', name='fc7'))
    model.add(Dropout(.5))
    model.add(Dense(487, activation='softmax', name='fc8'))

    model.load_weights('weights_C3D_sports1M_tf.h5')  
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(5, activation='softmax')) 
    for layer in model.layers[:3]:
      layer.trainable = False
    '''model_finetuned = Sequential()
    model_finetuned.add(model)
    model_finetuned.add(Dense(5, activation='softmax', name='fc9'))'''
    print(model.summary())



    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])

    history=model.fit(padded_videos, y_labels, epochs=3, batch_size=16, verbose=2)
    model.save("C3D_model.h5")
    print("model is saved")
    # print("model is loaded")

    # testing

    # if (os.path.exists("test_videos.npy")):
    #     testing_videos = np.load("test_videos.npy", allow_pickle=True)
    #     videos_names = np.load("videos_names.npy", allow_pickle=True)
    #     test_labels = np.load("test_labels.npy", allow_pickle=True)
    #
    # else:
    #     testing_videos, test_labels, videos_names = read_labeled_test_data()

    testing_videos, videos_names = read_test_data()

    predicted_labels = []

    for i in range(len(testing_videos)):
        padded_test = pad_sequences(testing_videos[i], maxlen=16, padding='pre')
        prediction = model.predict(padded_test)
        prediction = np.argmax(prediction, axis=1)
        # print(prediction)
        # majority_voting=multimode(prediction)
        majority_voting = stats.mode(prediction)[0]
        predicted_labels.append(majority_voting[0])

    # prediction = model.predict(padded_test_videos)
    # predicted_labels=np.argmax(prediction,axis=1)

    submitFile['Video'] = videos_names
    submitFile['Label'] = predicted_labels
    submitFile.to_csv('submitFile.csv', index=False)
    print("Submit File Saved Successfully")
    print(history.history['accuracy'])
    correct = 0
    # y_predicted = np_utils.to_categorical(predicted_labels)
    '''print("test_labels", len(test_labels), test_labels[1])
    print("predicted_labels", len(predicted_labels), predicted_labels[1])

    for i in range(len(test_labels)):
        # y_predicted[i, :] = np.where(y_predicted[i, :] == max(y_predicted[i, :]), 1, 0)
        if np.array_equal(predicted_labels[i], test_labels[i]):
            correct += 1

    print("Overall Accuracy", (correct / len(test_labels)) * 100, "%")
'''

main()
