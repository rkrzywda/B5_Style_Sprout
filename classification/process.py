import os
attributes = []
from PIL import Image

names = dict()
types = dict()
counter=1
rootdir = 'img'
clothes=[]
with open('list_category_img.txt', 'r') as file:
    for line in file:
        parts = line.split()
        types[parts[0]]=int(parts[1])-1
with open('list_category_cloth.txt', 'r') as file:
    for line in file:
        parts = line.split()
        clothes.append(parts[0])
csv = open('extra.csv', 'a')
for subdir, dirs, files in os.walk(rootdir):
    for f in files:
        prefix = ''
        if(f[:-17] not in names):
            prefix=str(counter)
            names[f[:-17]]=str(counter)
            counter+=1
        else:
            prefix=names[f[:-17]]
        newname = prefix+prefix+f[-12:]
        newname=newname[:-4]
        alternatename = 'img/'+f[:-17]+'/'+f[-16:]
        os.rename("img/"+f, 'fashion-dataset/images/'+newname+'.jpg')


'''
for subdir, dirs, files in os.walk(rootdir):
    for f in files:
        prefix = ''
        if(f[:-17] not in names):
            prefix=str(counter)
            names[f[:-17]]=str(counter)
            counter+=1
        else:
            prefix=names[f[:-17]]
        newname = prefix+prefix+f[-12:]
        newname=newname[:-4]
        alternatename = 'img/'+f[:-17]+'/'+f[-16:]
        try:
            csv.write(newname+','+clothes[types[alternatename]]+'\n')
        except:
            print("error")
'''

