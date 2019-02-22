Dependencies:
1. Ubuntu or any debian based linux OS
2. python 2.6 / 2.7
3. Tensorflow
4. protobuff

steps:
1. move to the server directory and run these commands on terminal
	
	protoc object_detection/protos/*.proto --python_out=.
	
	export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
run these commands on every terminal restart

2. change directory to object_detection
3. run command on terminal
	python object_detection_webcam.py
