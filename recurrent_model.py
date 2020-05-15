from tqdm import tqdm
import pandas as pd
from scipy import stats
import numpy as np
from keras.models import load_model
from statistics import mode  # ,multimode
import cv2 as cv
import tensorflow.keras.utils as np_utils
import os
from sklearn.utils import shuffle
# from random import shuffle
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, GRU, LSTM, Dropout, TimeDistributed, Dense, Bidirectional, Conv3D, Flatten, \
    MaxPooling3D, \
    ZeroPadding3D, BatchNormalization

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

                frame = cv.resize(frame, (200, 200))
                frame = frame.reshape(200, 200, 3)
                # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                one_video_frames = np.append(one_video_frames, frame)
                one_video_frames = np.reshape(one_video_frames, (-1, frame.shape[0], frame.shape[1], 3))

                if (leave_frame + 2 < video_frames):
                    leave_frame += 2
                    cap.set(1, leave_frame)
                # all_videos.append(np.array([one_video_frames, label]))

                if (sub_videos == 15):
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
    np.save('all_videosP1.npy', all_videos[:1000])
    np.save('all_videosP2.npy', all_videos[1000:2000])
    np.save('all_videosP3.npy', all_videos[2000:])
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

                frame = cv.resize(frame, (200, 200))
                frame = frame.reshape(200, 200, 3)
                # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

                one_video_frames = np.append(one_video_frames, frame)
                one_video_frames = np.reshape(one_video_frames, (-1, frame.shape[0], frame.shape[1], 3))

                if (sub_videos == 15):
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

            frame = cv.resize(frame, (200, 200))
            frame = frame.reshape(200, 200, 3)
            # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            one_video_frames = np.append(one_video_frames, frame)
            one_video_frames = np.reshape(one_video_frames, (-1, frame.shape[0], frame.shape[1], 3))
            if (sub_videos == 15):
                test_video.append(one_video_frames)
                one_video_frames = []
                sub_videos = 0
            sub_videos += 1
        testing_videos.append(test_video)
    np.save('testing_videos.npy', testing_videos)
    np.save('videos_names.npy', videos_names)
    return testing_videos, videos_names


def main():
    global all_videos
    global y_labels
    global max_no_frames
    save_path = 'all_videosP1.npy'

    if (os.path.exists(save_path)):
        all_videos = np.load(save_path, allow_pickle=True)
        x = np.load("all_videosP2.npy", allow_pickle=True)
        y = np.load("all_videosP3.npy", allow_pickle=True)
        all_videos = np.append(all_videos, x)
        all_videos = np.append(all_videos, y)
        y_labels = np.load('y_labels.npy', allow_pickle=True)
        print("train data loaded")
    else:
        read_data()
        print("train data created")

    # read_data()

    padded_videos = pad_sequences(all_videos, maxlen=15, padding='pre')
    # padded_videos=pad_sequences(all_videos,padding='pre')# bt3rf lw7dha el max lenght <3

    # model = load_model('recurrent_model.h5')

    # training model
    shape = (15, 200, 200, 3)
    model = Sequential()
    #    model.add(TimeDistributed(ZeroPadding3D(padding=(1, 2, 2), input_shape=shape)))
    model.add(Conv3D(32, kernel_size=(3, 5, 5), strides=(1, 2, 2), activation="relu", input_shape=shape))
    model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    #    model.add(TimeDistributed(ZeroPadding3D(padding=(1, 2, 2))))
    model.add(Conv3D(64, kernel_size=(3, 5, 5), strides=(1, 1, 1), activation="relu"))
    model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    #    model.add(TimeDistributed(ZeroPadding3D(padding=(1, 1, 1))))
    model.add(Conv3D(96, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation="relu"))
    model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(GRU(64, return_sequences=True), merge_mode='ave'))
    model.add(Bidirectional(GRU(64, return_sequences=True), merge_mode='ave'))

    model.add(Flatten())
    model.add((Dense(124, activation='relu')))
    model.add((Dense(5, activation='softmax')))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(padded_videos, y_labels, epochs=30, batch_size=15, verbose=2)
    model.save("recurrent_model.h5")
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

    testing_videos, test_labels, videos_names = read_labeled_test_data()

    predicted_labels = []

    for i in range(len(testing_videos)):
        padded_test = pad_sequences(testing_videos[i], maxlen=15, padding='pre')
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

    correct = 0
    # y_predicted = np_utils.to_categorical(predicted_labels)
    print("test_labels", len(test_labels), test_labels[1])
    print("predicted_labels", len(predicted_labels), predicted_labels[1])

    for i in range(len(test_labels)):
        # y_predicted[i, :] = np.where(y_predicted[i, :] == max(y_predicted[i, :]), 1, 0)
        if np.array_equal(predicted_labels[i], test_labels[i]):
            correct += 1

    print("Overall Accuracy", (correct / len(test_labels)) * 100, "%")


main()
