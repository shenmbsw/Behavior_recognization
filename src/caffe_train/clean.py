import os
for i in range(7):
    s = 'test'
    filelist = [ f for f in os.listdir('%s/%d/'%(s,i)) if f.endswith(".jpg") ]
    for j in filelist:
        os.remove("%s/%d/%s"%(s,i,j)) 

for i in range(7):
    s = 'train'
    filelist = [ f for f in os.listdir('%s/%d/'%(s,i)) if f.endswith(".jpg") ]
    for j in filelist:
        os.remove("%s/%d/%s"%(s,i,j)) 

for i in range(7):
    s = 'aug'
    filelist = [ f for f in os.listdir('%s/%d/'%(s,i)) if f.endswith(".jpg") ]
    for j in filelist:
        os.remove("%s/%d/%s"%(s,i,j)) 
