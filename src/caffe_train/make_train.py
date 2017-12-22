import os
#0:224,1:464,2:251,3:454,4:260,5:282
cwd = os.getcwd()
with open("train.txt", "w") as outF:
    for j in range(7):
        k = os.path.join(cwd,'aug/','%d'%j)
        ls = os.listdir(k)
        for img in ls:
            print('%s/%s %d'%(k,img,j))
            outF.writelines('%s/%s %d\r'%(k,img,j))

with open("test.txt", "w") as outF:
    for j in range(6):
        k = os.path.join(cwd,'test/','%d'%j)
        ls = os.listdir(k)
        for img in ls:
            print('%s/%s %d'%(k,img,j))
            outF.writelines('%s/%s %d\r'%(k,img,j))
