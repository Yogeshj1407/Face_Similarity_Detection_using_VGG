
import os
import pickle

actors = os.listdir('data')
# print(actors)

filename = []
for actor in actors:
    for file in os.listdir(os.path.join('data',actor)):
        filename.append(os.path.join('data',actor,file))

# print(len(filename))
pickle.dump(filename, open('filename.pkl','wb'))