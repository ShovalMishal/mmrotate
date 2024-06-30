# Docker instructions
1. Build the docker:
```shell
docker build -t oriented-rcnn:v1 -f docker/OOD_experiment_1/Dockerfile .
docker tag oriented-rcnn:v1 shovalmishal/oriented-rcnn:v1
docker push shovalmishal/oriented-rcnn:v1
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -v $(pwd):/workspace/ -it shovalmishal/ad-stage1:v1


```
On runai:
```shell
runai submit --name ood-supervised-experiment1-test -g 1.0 -i shovalmishal/oriented-rcnn:v1 --pvc=storage:/storage --large-shm 
```

