import os

rootdir = 'E:\Wenger\emotions'
count=0
for subdirs, dirs, files in os.walk(rootdir):
    print(subdirs)
    count+=1
    print(count)
    print(dirs)
    name = os.path.basename(subdirs)
    
