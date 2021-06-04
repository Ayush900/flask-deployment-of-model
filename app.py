from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import mne
from eeg_learn_functions import *
import pylsl

from pylsl import StreamInlet, resolve_stream
import time

import pandas as pd

import getopt

from random import random as rand

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer



# Define a flask app
app = Flask(__name__)


MODEL_PATH = 'models/deep_learning_model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# model.predict_classes()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

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

def channels_to_consider(raw):
    """
    For channels to consider , covering the functionality for channel independance
    """
    totalchannels = ['FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4',
                     'FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8','M1','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
                     'M2','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ','O2','CB2','HEOG','VEOG','EKG','FP1']
    
    uploaded_channels = raw.ch_names
    common_channels=[]
    for i in uploaded_channels:
        if i in totalchannels:
            common_channels.append(i)
    todrop = list(set(uploaded_channels) - set(common_channels))
    raw.drop_channels(ch_names=todrop)
    return raw

def channels_to_consider_realtime(df):
    """
    For channels to consider , covering the functionality for channel independance
    """

    totalchannels = ['FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4',
                     'FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8','M1','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
                     'M2','P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ','O2','CB2','HEOG','VEOG','EKG','FP1']
    uploaded_channels = list(df.columns)
    common_channels=[]
    for i in uploaded_channels:
        if i in totalchannels:
            common_channels.append(i)
    todrop = list(set(uploaded_channels) - set(common_channels))
    df.drop(todrop , axis='columns', inplace=True)
    return df

def clean_EEG(raw):
    raw=channels_to_consider(raw)
    if ['M1', 'M2'] in raw.ch_names:
        raw.set_eeg_reference(ref_channels=['M1', 'M2'])

#   Band Pass filtering
    raw.filter(0.5, 45, fir_design='firwin')
#   Resampling of data
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
    return df




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
    overlap = 0.5
    steps = make_steps(len(df),frame_duration,overlap)
    print(steps)
    for i,_ in enumerate(steps):
        frame = []
        if i == 0:
            continue
        else:
            for channel in df.columns:
                # print('check1:',steps[i][0])
                # print('check2:',steps[i][1])
                # print('check3:',int(channel))
                # brea
                snippet = np.array(df.iloc[int(steps[i][0]):int(steps[i][1]),int(channel)])
                f,Y =  get_fft(snippet)
                theta, alpha, delta = theta_alpha_delta_averages(f,Y)
                frame.append([theta, alpha, delta])

        frames.append(frame)
    return np.array(frames)

def make_data_pipeline(df,image_size,frame_duration,overlap):
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


    X_0 = make_frames(df,frame_duration)

    X_1 = X_0.reshape(len(X_0),len(list(df.columns))*3)
    locs_2d = [(-0.014727233440632792, 0.31252079949385286),
     (-0.0, 0.0),
     (0.014727233440632793, -0.3125207994938529),
     (0.09614589277335465, 0.3800464901961683),
     (-0.09614589277335465, -0.3800464901961683),
     (-0.08291018311929192, 0.9373008164913981),
     (0.18387846579147837, 0.8424432340679054),
     (0.32702972780435374, 0.6157461957014275),
     (0.26232493899144615, 0.31513166651955044),
     (0.0, 0.0),
     (-0.26256780986644307, -0.3149284549693126),
     (-0.3268030397919774, -0.6148877003257974),
     (-0.18471144213191895, -0.8422690383361868),
     (0.08278944286863567, -0.9365025605845323),
     (-0.12569922973128442, 1.2494167068562054),
     (0.34404468132670984, 1.1601284384909305),
     (0.6725732623205783, 0.8571531297685783),
     (0.6604609058211802, 0.42259024818521235),
     (0.0, 0.0),
     (-0.6604609058211802, -0.42259024818521235),
     (-0.6725732623205783, -0.8571531297685783),
     (-0.34404468132670984, -1.1601284384909305),
     (0.12569922973128442, -1.2494167068562054),
     (-0.16345617988131694, 1.5622685996751473),
     (0.4858722838371292, 1.4937634431432683),
     (1.0509921295981692, 1.1673973804129614),
     (1.4352860558774916, 0.6382437136987512),
     (0.0, 0.0),
     (-1.434693817427344, -0.639573881977754),
     (-1.0510697125390445, -1.1673275288690101),
     (-0.4858722838371292, -1.4937634431432683),
     (0.16345617988131694, -1.5622685996751473),
     (-0.18877737644985318, 1.876396605748076),
     (0.5491680801965788, 1.8518103662893555),
     (1.266757946772037, 1.6144048530691162),
     (1.9858032451854588, 1.2705991813194941),
     (2.8962871109801767, 1.2170149431537238),
     (-1.9858032451854588, -1.2705991813194941),
     (-1.266757946772037, -1.6144048530691162),
     (-0.5491680801965788, -1.8518103662893555),
     (0.18880917902585434, -1.8773554744380676),
     (-0.1939026952816692, 2.1920727680204823),
     (0.4860583794950023, 2.226887152908537),
     (1.1465646266580583, 2.158803151415301),
     (1.747585653914751, 2.099379424698052),
     (2.197754487150703, 2.244878530187838),
     (-1.7492069549204032, -2.0980296252418187),
     (-1.1475980398860457, -2.159233035571189),
     (-0.48825410694671384, -2.226398713448397),
     (0.1938572245617193, -2.1928857218900215),
     (-0.17632539031996594, 2.5089744161536354),
     (0.2740750738094674, 2.5941445563259284),
     (0.6743544337330677, 2.6655952562905805),
     (1.156989203732749, 2.9207842069442957),
     (-0.6743544337330677, -2.6655952562905805),
     (-0.2740750738094674, -2.5941445563259284),
     (0.17619116211795197, -2.5095748938615996),
     (-0.13315310595637417, 2.825589429020943),
     (-0.06601058765026968, 3.1408990756481536),
     (0.13315310595637417, -2.825589429020943)]
    images = gen_images(np.array(locs_2d),X_1, image_size, normalize=False)
    images = np.swapaxes(images, 1, 3)
    # print(len(images), ' frames generated with label ', labels[i], '.')
    # print('\n')
    # if i == 0:
    X = images
        # y = np.ones(len(images))*labels[0]
    # else:
    # X = np.concatenate((X,images),axis = 0)
        # y = np.concatenate((y,np.ones(len(images))*labels[i]),axis = 0)


    return X

def model_predict(file_path, model):
    # img = image.load_img(img_path, target_size=(224, 224))
    print(file_path)
    print(type(file_path))
    x = file_path.split('.')
    print(x[-1])
    if x[-1] == 'set':
        raw = mne.io.read_raw_eeglab(file_path,preload=True)
    elif x[-1] == 'vmrk':
        x[-1] = '.vhdr'
        file_path=''
        for i in x:
            file_path+=i
        raw = mne.io.read_raw_brainvision(file_path,preload=True)
    elif x[-1] == 'edf':
        raw = mne.io.read_raw_edf(file_path,preload=True)
    else:
        return -1

    # Preprocessing the raw_eeg_file
    # x = image.img_to_array(img)
    x = clean_EEG(raw)
    # x = np.true_divide(x, 255)
    # x = np.expand_dims(x, axis=0)
    image_size = 28                                                                               # 1 = current_mdd
    frame_duration = 1.0                                                                          # 2 = past_mdd
    overlap = 0.5
    images = make_data_pipeline(x,image_size,frame_duration,overlap)
    x = images
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='caffe')

    preds = model.predict_classes(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

def most_frequent(List):
    return max(set(List), key = List.count)

def startstream():
    srate = 100
    name = 'LSLExampleAmp'
    stream_type = 'EEG'
    channel_names = ['FP1','FPZ','FP2',
         # 'lal',
         'AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2',
         'FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8',
              # 'M1',
         'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
           # 'M2',
         'P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8',
         # 'CB1',
         'O1','OZ','O2',
         # 'CB2','HEOG','VEOG','Status' 
         ]
    n_channels = len(channel_names)
    help_string = 'SendData.py -s <sampling_rate> -n <stream_name> -t <stream_type>'
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hs:n:t:", longopts=["srate=", "name=", "type"])
    except getopt.GetoptError:
        print(help_string)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_string)
            sys.exit()
        elif opt in ("-s", "--srate"):
            srate = float(arg)
        elif opt in ("-n", "--name"):
            name = arg
        elif opt in ("-t", "--type"):
            stream_type = arg

    # first create a new stream info (here we set the name to BioSemi,
    # the content-type to EEG, 8 channels, 100 Hz, and float-valued data) The
    # last value would be the serial number of the device or some other more or
    # less locally unique identifier for the stream as far as available (you
    # could also omit it but interrupted connections wouldn't auto-recover).
    info = pylsl.StreamInfo(name, stream_type, n_channels, srate, 'float32', 'myuid2424')

    # append some meta-data
    info.desc().append_child_value("manufacturer", "LSLExampleAmp")
    chns = info.desc().append_child("channels")
    for label in channel_names:
        ch = chns.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")
    info.desc().append_child_value("manufacturer", "LSLExamples")
    cap = info.desc().append_child("cap")
    cap.append_child_value("name", "ComfyCap")
    cap.append_child_value("size", "54")
    cap.append_child_value("labelscheme", "10-20")

    # next make an outlet; we set the transmission chunk size to 32 samples and
    # the outgoing buffer size to 360 seconds (max.)
    outlet = pylsl.StreamOutlet(info, 32, 1)

    if False:
        # It's unnecessary to check the info when the stream was created in the same scope; just use info.
        # Use this code only as a sanity check if you think something when wrong during stream creation.
        check_info = outlet.get_info()
        assert check_info.name() == name
        assert check_info.type() == stream_type
        assert check_info.channel_count() == len(channel_names)
        assert check_info.channel_format() == pylsl.cf_float32
        assert check_info.nominal_srate() == srate

    print("now sending data...")
    start_time = pylsl.local_clock()
    sent_samples = 0
    while True:
        elapsed_time = pylsl.local_clock() - start_time
        required_samples = int(srate * elapsed_time) - sent_samples
        if required_samples > 0:
            # make a chunk==array of length required_samples, where each element in the array
            # is a new random n_channels sample vector
            mychunk = [[rand() for chan_ix in range(n_channels)]
                       for samp_ix in range(required_samples)]
            # get a time stamp in seconds (we pretend that our samples are actually
            # 125ms old, e.g., as if coming from some external hardware)
            stamp = pylsl.local_clock() - 0.125
            # now send it and wait for a bit
            # Note that even though `rand()` returns a 64-bit value, the `push_chunk` method
            #  will convert it to c_float before passing the data to liblsl.
            outlet.push_chunk(mychunk, stamp)
            sent_samples += required_samples
        time.sleep(0.02)
    return


@app.route('/get_realtime_eeg_data',methods=['GET'])
def get_realtime_eeg_data():
    startstream()



    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    start_time = time.time()
    seconds = 15

    data=[]

    while True:

        chunk, timestamps = inlet.pull_chunk()
        if timestamps:
            for i in chunk:
                data.append(i)

        current_time = time.time()
        elapsed_time = current_time - start_time

        if elapsed_time > seconds:
            print("Finished iterating in: " + str(int(elapsed_time))  + " seconds")
            break

    index=[]
    for i in range(len(data)):
        index.append(str(i*4))

    info = inlet.info()

    ch = info.desc().child("channels").child("channel")

    channels=[]

    for k in range(info.channel_count()):
        channels.append(ch.child_value("label"))
        ch = ch.next_sibling()

    df = pd.DataFrame(data, columns = channels , index=index)

    df=channels_to_consider_realtime(df)

    channels_remain=df.columns

    x=list(df.columns)

    for i in range(len(x)):
        x[i]=i

    df.columns = x

    image_size = 28                                                                               # 1 = current_mdd
    frame_duration = 1.0                                                                          # 2 = past_mdd
    overlap = 0.5
    images = make_data_pipeline(df,image_size,frame_duration,overlap)
    x = images

    print(x)

    result = model.predict_classes(x)

    result = result.tolist()
    ans = most_frequent(result)
    if ans == 0:
        result = "The Subject is Controlled !"
    elif ans == 1:
        result = "The Subject is currently suffering from MDD !"
    else:
        result = "The Subject had suffered from MDD in the past !"

    return result




@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files.getlist('file')

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        for file in f:
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(file.filename))
            file.save(file_path)
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        # file_path = os.path.join(
        #     basepath, 'uploads', secure_filename(f.filename))
        # f.save(file_path)

        # Make prediction
        # name = f[1].filename
        # print(f[])
        result = model_predict(file_path, model)
        if type(result) == int and result == -1:
            return "Format of the file uploaded is not compatible ! You can only upload file of formats : .set or .vhdr or .edf"
        result = result.tolist()
        ans = most_frequent(result)
        if ans == 0:
            result = "The Subject is Controlled."
        elif ans == 1:
            result = "The Subject is currently suffering from Depression"
        else:
            result = "The Subject had suffered from Depression in the past."
        # print(ans)
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        return result

    return None


if __name__ == '__main__':
    app.run(debug=True)
