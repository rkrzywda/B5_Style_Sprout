import os

'''
#add clothing type tag to each image
file1 = open('list_category_cloth.txt', 'r')
lines = file1.readlines()
categories = []
for line in lines:
    parts = line.split(" ")
    name = parts[0]
    categories.append(name)
categories.pop(0)
categories.pop(0)
print(categories)

file2 = open('list_category_img.txt', 'r')
lines = file2.readlines()

file3 = open('labels.csv', 'a')

for line in lines:
    line = line[4:]
    line = line[:-1]
    slash = line.find("/")
    line = line[:slash]+"_"+line[slash+1:]
    parts = line.split(" ")
    name = parts[0]
    clothing_type = int(parts[-1])
    file3.write(name + ", "+ categories[clothing_type-1]+"\n")


#renaming images
rootdir = '../../../../img/'
for subdir, dirs, files in os.walk(rootdir):
    print(subdir)
    if(subdir=='../../../../img/'):
        continue
    for f in files:
        os.rename(subdir+"/"+f, subdir[16:]+"_"+f)
'''

