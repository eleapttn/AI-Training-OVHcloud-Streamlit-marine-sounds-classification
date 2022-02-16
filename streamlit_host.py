import librosa
import streamlit as st
import csv
import os
import numpy as np
import pandas as pd
import keras
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from PIL import Image



def save_file(sound_file):
    
    with open(os.path.join("test_data", sound_file.name),"wb") as f:
         f.write(sound_file.getbuffer())
    
    return sound_file.name



def prediction(dataframe):
    
    df = pd.read_csv('data.csv')
    class_list = df.iloc[:,-1]
    
    converter = LabelEncoder()
    y = converter.fit_transform(class_list)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(df.iloc[:, 1:27]))
    X_test = scaler.transform(np.array(df_test.iloc[:, 1:27]))

    model = load_model('saved_model/my_model')
    
    # Check its architecture
    model.summary()
    
    predictions = model.predict(X_test)
    classes = np.argmax(predictions, axis = 1)
    result = converter.inverse_transform(classes)
    
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    
    return result



if __name__ == '__main__':
    
    st.image(Image.open('logo_ovh.png'), width=200)
    
    st.write('___')
    
    st.sidebar.title('Marine mammal sounds classification')
    select = st.sidebar.selectbox('', ['Marine mammals', 'Prediction'], key='1')
    st.sidebar.write(select)
    
    if select=='Prediction':
        
        st.write('# Prediction')
        st.write('### Choose a marine mammal sound file in .wav format')
        
        uploaded_file = st.file_uploader(" ", type='wav')
        
        if uploaded_file is not None:
                
            #st.write(file)
            file_details = {'filename':uploaded_file.name, 'filetype':uploaded_file.type, 'filesize':uploaded_file.size}
            st.write(file_details)
            
            st.write('### Play audio')
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes, format='audio/wav')
    
            save_file(uploaded_file)
        
            header_test = 'filename length chroma_stft_mean chroma_stft_var rms_mean rms_var spectral_centroid_mean spectral_centroid_var spectral_bandwidth_mean \
            spectral_bandwidth_var rolloff_mean rolloff_var zero_crossing_rate_mean zero_crossing_rate_var harmony_mean harmony_var perceptr_mean perceptr_var tempo mfcc1_mean mfcc1_var mfcc2_mean \
            mfcc2_var mfcc3_mean mfcc3_var mfcc4_mean mfcc4_var'.split()
            
            csv_file = open('test.csv', 'w', newline = '')
            with csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(header_test)
            
            for filename in os.listdir('test_data'):
                sound = f'test_data/{filename}'
                y, sr = librosa.load(sound, mono = True, duration = 30)
                chroma_stft = librosa.feature.chroma_stft(y = y, sr = sr)
                rmse = librosa.feature.rms(y = y)
                spec_cent = librosa.feature.spectral_centroid(y = y, sr = sr)
                spec_bw = librosa.feature.spectral_bandwidth(y = y, sr = sr)
                rolloff = librosa.feature.spectral_rolloff(y = y, sr = sr)
                zcr = librosa.feature.zero_crossing_rate(y)
                mfcc = librosa.feature.mfcc(y = y, sr = sr)
                to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
                
                for e in mfcc:
                    to_append += f' {np.mean(e)}'
                
                csv_file = open('test.csv', 'a', newline = '')
                with csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(to_append.split())
            
                os.remove(f'test_data/{filename}')

                df_test = pd.read_csv('test.csv')
            
            st.write('### Classification results')
            
            if st.button('Predict'):
                st.write("The marine mammal is: ", str(prediction(df_test)).replace('[', '').replace(']', '').replace("'", '').replace('"', ''))
        
        else:
            st.write('The file has not been uploaded yet')
    
    
    else:
        st.write('# Marine mammals')
        st.write('The different marine mammals studied are the following.')
        st.write('For more information, please refer to this [link](https://cis.whoi.edu/science/B/whalesounds/index.cfm).')
        st.image(Image.open('marine_mammal_animals.png'))
        
    
    