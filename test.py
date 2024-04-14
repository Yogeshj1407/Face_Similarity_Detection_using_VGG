from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
import cv2
from mtcnn import MTCNN
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

feature_list = np.array(pickle.load(open('embedding.pkl','rb')))
filename = pickle.load(open('filename.pkl','rb'))

model_weights_path = "rcmalli_vggface_tf_notop_resnet50.h5"
model = VGGFace(model='resnet50',input_shape=(244,244,3),include_top=False,pooling='avg',weights=None)
model.load_weights(model_weights_path)

detector = MTCNN()

sample_img = cv2.imread('sample/ab.jpg')
result = detector.detect_faces(sample_img)

x, y, width, height = result[0]['box']

face = sample_img[y:y+height, x:x+width]

#feature extraction
image = Image.fromarray(face)
image = image.resize((244,244))

face_array = np.asarray(image)
face_array = face_array.astype('float32')
expanded_img = np.expand_dims(face_array,axis=0)

preprocess_img = preprocess_input(expanded_img)

result = model.predict(preprocess_img).flatten()

# print(result)
# print(result.shape)

#comparing result with cosine similarity with all images
similarity = []
for i in range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1,-1),feature_list[i].reshape(1,-1))[0][0])
# print(len(similarity))

index_pos = sorted(list(enumerate(similarity)), reverse= True, key= lambda x : x[1])[0][0]

temp_img = cv2.imread(filename[index_pos])
cv2.imshow('output', temp_img)
cv2.waitKey(0)

