import os.path
import streamlit as st
from PIL import Image
import cv2
from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity


detector = MTCNN()

feature_list = pickle.load(open('embedding.pkl','rb'))
filename = pickle.load(open('filename.pkl','rb'))

model_weights_path = "rcmalli_vggface_tf_notop_resnet50.h5"
model = VGGFace(model='resnet50',input_shape=(244,244,3),include_top=False,pooling='avg',weights=None)
model.load_weights(model_weights_path)



def save_uploaded_image(uploaded_img):
    try:
        with open(os.path.join('uploads',uploaded_img.name),'wb') as f:
            f.write(uploaded_img.getbuffer())
        return True
    except:
        return False



def extract_feature(img_path,model,detector):
    img = cv2.imread(img_path)
    result = detector.detect_faces(img)

    if result:
        x, y, width, height = result[0]['box']

        face = img[y:y + height, x:x + width]

        # feature extraction
        image = Image.fromarray(face)
        image = image.resize((244, 244))

        face_array = np.asarray(image)
        face_array = face_array.astype('float32')
        expanded_img = np.expand_dims(face_array, axis=0)

        preprocess_img = preprocess_input(expanded_img)

        result = model.predict(preprocess_img).flatten()
        return result

    else:
        print('No Face detected')
        return None



def recommend(feature_list,feature):
    similarity = []
    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(feature.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])

    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos



st.title('Face similarity Detection using VGG')

uploaded_img = st.file_uploader('Choose an Image :')



if uploaded_img != None:
    if save_uploaded_image(uploaded_img):

        #display image
        display_img = Image.open(uploaded_img)

        #extract features
        feature = extract_feature(os.path.join('uploads',uploaded_img.name),model,detector)
        if feature is not None:
            pass
        else:
            st.text('No Face detected')

        #recommend
        index_pos = recommend(feature_list,feature)
        pred_actor = " ".join(filename[index_pos].split('\\')[1].split('_'))
        # st.text(index_pos)

        #display
        col1, col2 = st.columns(2)
        with col1:
            st.header('Your Uploaded Image')
            st.image(display_img)
        with col2:
            st.header("Seems like " + pred_actor)
            st.image(filename[index_pos],width=300)





