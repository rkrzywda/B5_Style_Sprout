import os
attributes = []

file5 = open('list_attr_cloth.txt', 'r')
lines = file5.readlines()
for line in lines:
    parts = line.split(" ")
    name = parts[0]
    attributes.append(name)
attributes.pop(0)
attributes.pop(0)

labels = dict()

file1 = open('list_category_cloth.txt', 'r')
lines = file1.readlines()
categories = []
for line in lines:
    parts = line.split(" ")
    name = parts[0]
    categories.append(name)
categories.pop(0)
categories.pop(0)

file2 = open('list_category_img.txt', 'r')
lines = file2.readlines()

for line in lines:
    line = line[4:]
    line = line[:-1]
    slash = line.find("/")
    line = line[:slash]+"_"+line[slash+1:]
    parts = line.split(" ")
    name = parts[0]
    clothing_type = int(parts[-1])
    labels[name]=[categories[clothing_type-1]]

file4 = open('list_attr_img.txt', 'r')
lines = file4.readlines()

for line in lines:
    split = line.split(" ")
    split = [i for i in split if i]
    indices = split[-1000:]
    last = indices.pop()
    last = last[:-1]
    indices.append(last)
    name = split[0]
    name = name[4:]
    slash = name.find("/")
    name = name[:slash]+"_"+name[slash+1:]
    has = []
    for i in range(1000):
        if indices[i]=="1":
            has.append(attributes[i])
    labels[name]+=has

file6 = open('labels.csv', 'a')
for key in labels:
    line = key + " ,"+ 


'''
#renaming images
rootdir = '../../../../img/'
for subdir, dirs, files in os.walk(rootdir):
    print(subdir)
    if(subdir=='../../../../img/'):
        continue
    for f in files:
        os.rename(subdir+"/"+f, subdir[16:]+"_"+f)
'''

