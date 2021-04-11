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


### Instructions

The file Preprocessing_on_raw_EEG_using_MNE_package.ipynb consists the code for the Preprocessing of the EEG data using the mne package pf python .
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
