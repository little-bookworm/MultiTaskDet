export LD_LIBRARY_PATH=/work/Tensorrt8.2.1/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/work/Tensorrt8.2.1::$LIBRARY_PATH
#if tensorrt 8.xx
./trtexec --onnx=/hostdata/projects/parking_perception/modules/MultiTaskDet/model/yolop-800-800-all-5th.onnx --saveEngine=/hostdata/projects/parking_perception/modules/MultiTaskDet/model/yolop-800-800-all-5th.trt --workspace=2048 --iterations=1 --fp16
