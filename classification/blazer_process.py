import pandas as pd
import os
directory = 'fashion-dataset/images'

counter = 0
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if('blazer' in f):
        os.rename(f, f[:23]+'8888'+f[30:])
    continue
    if os.path.isfile(f):
        if(f[-3:]=='jpg'):
            new_name = 'archive/images_compressed/blazer_'+str(counter)+".jpg"
            os.rename(f, new_name)
            counter+=1

'''
f = open("./archive/blazers.csv", "w")
for counter in range(109):
    f.write('blazer_'+str(counter)+',Blazers,\n')
f = open("./archive/blazers.csv", "r")
print(f.read())

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
df = pd.read_csv('archive/images.csv')
df = df[df['label'] == 'Blazer']
df = df[['image', 'label']] 
correct = set()
for img in df['image']:
    correct.add(img)
'''

