protoc *.proto --python_out=.
echo "protoc object_detection/protos/*.proto --python_out=."
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
