def channels_to_consider(raw):
    """
    For channels to consider , covering the functionality for channel independence
    Any EEG data following the main 10-20 EEG system having channel number 
    less than or equal to 67 channels can be uploaded
    """
    totalchannels = ['FPZ',
                     'FP2',
                     'AF3',
                     'AF4',
                     'F7',
                     'F5',
                     'F3',
                     'F1',
                     'FZ',
                     'F2',
                     'F4',
                     'F6',
                     'F8',
                     'FT7',
                     'FC5',
                     'FC3',
                     'FC1',
                     'FCZ',
                     'FC2',
                     'FC4',
                     'FC6',
                     'FT8',
                     'T7',
                     'C5',
                     'C3',
                     'C1',
                     'CZ',
                     'C2',
                     'C4',
                     'C6',
                     'T8',
                     'M1',
                     'TP7',
                     'CP5',
                     'CP3',
                     'CP1',
                     'CPZ',
                     'CP2',
                     'CP4',
                     'CP6',
                     'TP8',
                     'M2',
                     'P7',
                     'P5',
                     'P3',
                     'P1',
                     'PZ',
                     'P2',
                     'P4',
                     'P6',
                     'P8',
                     'PO7',
                     'PO5',
                     'PO3',
                     'POZ',
                     'PO4',
                     'PO6',
                     'PO8',
                     'CB1',
                     'O1',
                     'OZ',
                     'O2',
                     'CB2',
                     'HEOG',
                     'VEOG',
                     'EKG',
                     'FP1']
    uploaded_channels = raw.ch_names
    common_channels=[]
    for i in uploaded_channels:
        if i in totalchannels:
            common_channels.append(i)
    todrop = list(set(uploaded_channels) - set(common_channels))
    raw.drop_channels(ch_names=todrop)
    return raw
