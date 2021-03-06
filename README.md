# Deployment-Deep-Learning-Model


### Objective 
Make an algorithm based system where 
depression can be detected from resting state 
EEG rather than questionnaire.


### Dataset worked on
http://predict.cs.unm.edu/downloads.php 
Accession: d003

### Research papers and reference materials
https://docs.google.com/spreadsheets/d/1Ltiheuj3ifW_92J-VCAF3AYsaIYeyM_9pnoIZfc-_DM/edit#gid=0


### Instructions For the codes found in different files

The file Preprocessing_on_raw_EEG_using_MNE_package.ipynb consists the code for the preprocessing of the EEG data using the MNE package of Python .
The preprocessing steps taken are :
1.Rereferencing using the mastoid electrodes (M1,M2)
2.Band Pass Filtering
3.Downsampling
4.Removal of artifacts
5.Remove Bad Channels
6.Run ICA


The model.py file consist the code for the approach to train the model which would be used to make predictions .
We have used the CNN model for the training of our model , as the data which is to be used to train the model
eventually are images , for which the CNN is best suited.



The rest of the files contribute to the flask structure for the deployment of our model .

### Brief Explaination of Approach and what has been achieved so far

For the deployment of our model we have prefered the
Python's framework flask . Having finished training our
model we saved our model as a .h5 extension file and imported it
on our backend.
From the frontend when we upload files of different formats
on which we have given the prediction, our backend
first cleans up our file to remove all the artifacts and the bad
channels from the raw EEG by applying all the preprocessing steps and then converts it into a dataframe which
contains the time series data , which is further converted to
frequency band data by applying FFT . The spacial images are
formed which are in turn fed into our model.We have fed theta,alpha and delta band images and the prediction
is made using the deep learning model which would then be displayed on the screen.


### Files which could be used for prediction

##### Type of EEG data to be uploaded    ||      Number of files to be uploaded    ||        Extension of files to be uploaded
#####  EEGlab                             ||     2                                  ||       .set and .fdt
##### BrainVision                        ||      3                                 ||       .vhdr , .vmrk and .dat
#####  European data format               ||      1                                 ||        .edf








## How to run the model
1. Download ALL the repository files as a zip folder
2. Extract and Open the command prompt for this folder
3. Type the command "python app.py" in the command prompt
4. Install all the necessary system requirements (mne, flask, tensorflow etc) as per the need
5. Once the command "python app.py" is successful , copy the link address generated and open it in your web browser to see the model running.
