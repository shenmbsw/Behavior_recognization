## Final project for EC720: ICPR 2010 Contest on Semantic Description of Human Activities

Copyright <2017> Shen Shen

the data base and the task is provided at:
http://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html#Data

The challenge require us to recognize ongoing human activities from continuous videos.

The thesis Large-scale Video Classification with Convolutional Neural Networks written by Andrej Karpathy provided me the inspiration of how to design a CNN network to do the behavior of a frame by aquire the information of not only one frame but also frames in a time span.

### The challenging part of this project is:

1. How to design a CNN network to receive a converge and effective result. Intuitivly, we may think that frames in time span would provide a more confident predict result. Thus, I want to make a comparation between single frame CNN and sequence frame CNN.

2. How to do the pre processing of the video using motion detection method. I will try both continuous frames subtraction, background subtraction, and no preprocessing video as the input to find if the motion detection method could provide a better result.

### Executing
1. Down load data set and save in the master root.
2. Execute src/first.py
 
 3D CNN is trained by tensorflow, so the data should be provide in batch style.
 seq_data_generators read the numpy array of each video segment and provide them as a dictionary to feed in main_3dconv.py. main_3dconv.py produce the trained model and it provide training loss and validation accuracy in a tensorflow graph file. Use tensor broad to see the training log.
 The my_script.sh is for runing all those code on SCC.
 

 2D CNN is trained by caffe, the modify the leave k out.sh file to produce leave k out dataset.
 all the decode frame should be labeled and saperate in different folder in caffe_train/all. after runing the script, a train.txt indicating the path of all train file and a test.txt file will be generated.
