# Use of MultiTaskDet

## 1 Requirement
This codebase has been developed with cuda-11.1, TensorRT-8.4.1.5, OpenCV-3 and YAML ...

## 2 Installation
### 2.1 Create docker container
`docker run --gpus all -it -v /mnt/Data/Dataset:/hostdata/Dataset -v /mnt/work/projects:/hostdata/projects -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY -e GDK_SCALE -e GDK_DPI_SCALE --net=host --privileged=true --name=perception_20231204 ubuntu18.04_cu11.1_trt8.4:20231204 /bin/bash`

### 2.2 Install modules *MultiTaskDet*
`
cd MultiTaskDet
bash build.sh
`

## 3 Test
### 3.1 Run test
`
bash run_test.sh
`
### 3.2 Results Visualization
<div align=left><img src="./test/out_det.jpg" width=300 height=300>
<div align=left><img src="./test/out_da.jpg" width=300 height=300>
<div align=left><img src="./test/out_ll.jpg" width=300 height=300>


