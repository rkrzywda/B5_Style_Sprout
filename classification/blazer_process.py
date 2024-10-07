import pandas as pd

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
df = pd.read_csv('archive/images.csv')
df = df[df['label'] == 'Blazer']
df = df[['image', 'label']] 
correct = set()
for img in df['image']:
    correct.add(img)

import os
directory = '.'

counter = 0
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):
        if(f[-3:]=='jpg'):
            new_name = 'archive/images_compressed/blazer_'+str(counter)+".jpg"
            os.rename(f, new_name)
            counter+=1
