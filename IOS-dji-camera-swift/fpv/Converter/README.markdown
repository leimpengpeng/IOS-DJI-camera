If you want to convert the pre-trained models by yourself. Here is the instructions:

1. Download the pre-trained darknet models.

`% curl https://pjreddie.com/media/files/yolo.weights > yolo.weights`

2. Also download the configuration file in [darknet's repo](https://github.com/pjreddie/darknet/blob/master/cfg/yolo.cfg).

3. Build and install [darkflow](https://github.com/thtrieu/darkflow). And use the following command to convert the weights to tensorflow pb file:

`% flow --model yolo.cfg --load yolo.weights --savepb`

4. Install [tf-coreml](https://github.com/tf-coreml/tf-coreml).

5. Launch Jupyter notebook and open _Convert_pb_coreml.ipynb_, and run the script. After it finishes you will get yolo.mlmodel. Copy the file to Xcode project. 

