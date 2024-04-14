'''
mtcnn==0.1.0
tensorflow==2.3.1
keras==2.4.3
keras-vggface==0.6
keras-applications==1.0.8
'''

'''
# Getting filenames in Pickle file

import os
import pickle

actors = os.listdir('data')

filename = []
for actor in actors:
    for file in os.listdir(os.path.join('data',actor)):
        filename.append(os.path.join('data',actor,file))

pickle.dump(filename, open('filenames.pkl','wb'))
'''

from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm

filename = pickle.load(open('filename.pkl','rb'))
# Provide the local file path where you saved the downloaded model weights
model_weights_path = "rcmalli_vggface_tf_notop_resnet50.h5"

model = VGGFace(model='resnet50',input_shape=(244,244,3),include_top=False,pooling='avg',weights=None)
model.load_weights(model_weights_path)

print(model.summary())

def feature_extractor(img_path,model):
    img = image.load_img(img_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    result = model.predict(preprocessed_img).flatten()

    return result

features = []

for file in tqdm(filename):
    features.append(feature_extractor(file,model))

pickle.dump(features,open("embedding.pkl","wb"))






