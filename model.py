import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as scs
import scipy.misc
import re
import os
from numpy import genfromtxt
import mne
import math as m
from eeg_learn_functions import *

df = pd.read_csv('C:/Users/ayush/OneDrive/Desktop/channels.csv')

def extract_electrode_locations(df):
    l1 = list(df.columns)
    channels = []
    channels.append(l1[0].split()[0])
    cn =  list(df['     FP1'])
    for i in cn:
        channels.append(i.split()[0])
    x_coordinate = []
    y_coordinate = []
    z_coordinate = []
    x_coordinate.append(l1[1])
    y_coordinate.append(l1[2])
    z_coordinate.append(l1[3])
    x_coordinate.extend(list(df[l1[1]]))
    y_coordinate.extend(list(df[l1[2]]))
    z_coordinate.extend(list(df[l1[3]]))
    for i in range(len(x_coordinate)):
        x_coordinate[i] = float(x_coordinate[i])
        y_coordinate[i] = float(y_coordinate[i])
        z_coordinate[i] = float(z_coordinate[i])
    electrode_location = {}
    for i in range(len(channels)):
        x = x_coordinate[i]
        y = y_coordinate[i]
        z = z_coordinate[i]
        tu = (x,y,z)
        electrode_location[channels[i]] = tu
    bad_channels=['CB1', 'CB2', 'HEOG', 'VEOG', 'EKG','M1','M2']
    for i in list(electrode_location.keys()):
        for j in bad_channels:
            if i==j:
                electrode_location.pop(i,None)
    return list(electrode_location.values())

locs_3d = extract_electrode_locations(df)
all_set_files = os.listdir("C:/Users/ayush/OneDrive/Desktop/all set files")
all_set_files1 = []

for i in range(len(all_set_files)):
    if i%2!=0:
        all_set_files1.append(all_set_files[i])

def preprocessing_of_rawEEG(all_set_files):
    all_files_to_df = []
    for i, file in enumerate(all_set_files):
#       Loading the .set file using the mne package
        raw = mne.io.read_raw_eeglab('C:/Users/ayush/OneDrive/Desktop/all set files/'+file, preload=True)
#       Rerefercing the data using the mastoid electrodes
        raw.set_eeg_reference(ref_channels=['M1', 'M2'])
#       Band Pass filtering
        raw.filter(0.5, 45, fir_design='firwin')
#       Resampling of data
        raw.resample(250, npad="auto")
#       Removal of bad channels
        bad_channels=['CB1', 'CB2', 'HEOG', 'VEOG', 'EKG','M1','M2']
        x=raw.ch_names
        channels_to_remove = []
        for i in x:
            for j in bad_channels:
                if i==j:
                    channels_to_remove.append(i)
        raw.drop_channels(ch_names=channels_to_remove)
        raw_tmp = raw.copy()
        raw_tmp.filter(1, None)
#       Run ICA on the data
        ica = mne.preprocessing.ICA(method="fastica",random_state=1)
        ica.fit(raw_tmp)
        picks = len(raw.ch_names)
        ica.exclude = [0,1]
        raw_corrected = raw.copy()
        ica.apply(raw_corrected)
#       Removal of Artifacts
        raw.del_proj()
#       Saving the data to dataframes
        df = raw.to_data_frame()
        df=df.set_index('time')
        numberofelectrodes = len(df.columns)
        electrodes = list(df.columns)
        res = {}
        j=0
        for i in electrodes:
            res[i]=j
            j+=1
        df=df.rename(columns=res)
        all_files_to_df.append(df)
    return all_files_to_df

dataframes = preprocessing_of_rawEEG(all_set_files1)

delta = (0,4)
theta = (4,8)
alpha = (8,12)
# beta = (12,40)

def theta_alpha_delta_averages(f,Y):
    theta_range = (4,8)
    alpha_range = (8,12)
#     beta_range = (12,40)
    delta_range = (0,4)
    theta = Y[(f>theta_range[0]) & (f<=theta_range[1])].mean()
    alpha = Y[(f>alpha_range[0]) & (f<=alpha_range[1])].mean()
#     beta = Y[(f>beta_range[0]) & (f<=beta_range[1])].mean()
    delta = Y[(f>delta_range[0]) & (f<=delta_range[1])].mean()
    return theta, alpha, delta
def get_fft(snippet):
    Fs = 500.0;  # sampling rate
    #Ts = len(snippet)/Fs/Fs; # sampling interval
    snippet_time = len(snippet)/Fs
    Ts = 1.0/Fs; # sampling interval
    t = np.arange(0,snippet_time,Ts) # time vector

    # ff = 5;   # frequency of the signal
    # y = np.sin(2*np.pi*ff*t)
    y = snippet
#     print('Ts: ',Ts)
#     print(t)
#     print(y.shape)
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(n//2)] # one side frequency range

    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[range(n//2)]
    #Added in: (To remove bias.)
    #Y[0] = 0
    return frq,abs(Y)
#f,Y = get_fft(np.hanning(len(snippet))*snippet)

def make_steps(samples,frame_duration,overlap):
    '''
    in:
    samples - number of samples in the session
    frame_duration - frame duration in seconds
    overlap - float fraction of frame to overlap in range (0,1)

    out: list of tuple ranges
    '''
    #steps = np.arange(0,len(df),frame_length)
    Fs = 500
    i = 0
    intervals = []
    samples_per_frame = Fs * frame_duration
    while i+samples_per_frame <= samples:
        intervals.append((i,i+samples_per_frame))
        i = i + samples_per_frame - int(samples_per_frame*overlap)
    return intervals

def make_frames(df,frame_duration):
    '''
    in: dataframe or array with all channels, frame duration in seconds
    out: array of theta, alpha, beta averages for each probe for each time step
        shape: (n-frames,m-probes,k-brainwave bands)
    '''
    Fs = 500.0
    frame_length = Fs*frame_duration
    frames = []
    steps = make_steps(len(df),frame_duration,overlap)
    for i,_ in enumerate(steps):
        frame = []
        if i == 0:
            continue
        else:
            for channel in df.columns:
                snippet = np.array(df.loc[steps[i][0]:steps[i][1],int(channel)])
                f,Y =  get_fft(snippet)
                theta, alpha, delta = theta_alpha_delta_averages(f,Y)
                frame.append([theta, alpha, delta])

        frames.append(frame)
    return np.array(frames)

def cart2sph(x,y,z):                                               ## cart2sph(X,Y,Z) transforms Cartesian coordinates stored in corresponding elements of arrays X , Y , and Z into spherical coordinates
    XsqPlusYsq = x**2 + y**2
    r = m.sqrt(XsqPlusYsq + z**2)               # r
    elev = m.atan2(z,m.sqrt(XsqPlusYsq))     # theta
    az = m.atan2(y,x)                           # phi
    return r, elev, az

def pol2cart(rho, phi):                                            ## pol2cart(THETA,RHO) transforms the polar coordinate data stored in corresponding elements of THETA and RHO to two-dimensional Cartesian, or xy, coordinates.
    x = rho * np.cos(phi)                                          ## The arrays THETA and RHO must be the same size (or either can be scalar). The values in THETA must be in radians.
    y = rho * np.sin(phi)
    return(x, y)

def azim_proj(pos):                                               ## Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])              ## Imagine a plane being placed against (tangent to) a globe. If
    return pol2cart(az, m.pi / 2 - elev)                          ## a light source inside the globe projects the graticule onto
                                                                  ## the plane the result would be a planar, or azimuthal, map projection.
locs_2d=[]
for i in range(len(locs_3d)):
    locs_2d.append(azim_proj(locs_3d[i]))

def make_data_pipeline(dataframes,labels,image_size,frame_duration,overlap):
    '''
    IN:
    file_names - list of strings for each input file (one for each subject)
    labels - list of labels for each
    image_size - int size of output images in form (x, x)
    frame_duration - time length of each frame (seconds)
    overlap - float fraction of frame to overlap in range (0,1)

    OUT:
    X: np array of frames (unshuffled)
    y: np array of label for each frame (1 or 0)
    '''
    ##################################
    ###Still need to do the overlap###!!!
    ##################################

    Fs = 500.0   #sampling rate
    frame_length = Fs * frame_duration

    for i, df in enumerate(dataframes):
        X_0 = make_frames(df,frame_duration)

        X_1 = X_0.reshape(len(X_0),60*3)

        images = gen_images(np.array(locs_2d),X_1, image_size, normalize=False)
        images = np.swapaxes(images, 1, 3)
        print(len(images), ' frames generated with label ', labels[i], '.')
        print('\n')
        if i == 0:
            X = images
            y = np.ones(len(images))*labels[0]
        else:
            X = np.concatenate((X,images),axis = 0)
            y = np.concatenate((y,np.ones(len(images))*labels[i]),axis = 0)


    return X,np.array(y)

labels1 = [0,1,1,1,1,0,1,1,2,2,2,2,0,0,0,2,0,2,0,2,2,2,2,0,1,1,2,1,2,1,]                       # 0 = controlled
image_size = 28                                                                               # 1 = current_mdd
frame_duration = 1.0                                                                          # 2 = past_mdd
overlap = 0.5
X, y = make_data_pipeline(dataframes,labels1,image_size,frame_duration,overlap)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25,shuffle=True)
# input image dimensions
img_rows, img_cols = 28, 28

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

input_shape = (img_rows, img_cols, 3)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

batch_size = 32
num_classes = 3
epochs = 100

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(10))
model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#x_train /= 255
#x_test /= 255


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)
