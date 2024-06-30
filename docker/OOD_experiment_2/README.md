# Docker instructions
1. Build the docker:
```shell
docker build -t oriented-rcnn:v2 -f docker/OOD_experiment_2/Dockerfile .
docker tag oriented-rcnn:v2 shovalmishal/oriented-rcnn:v2
docker push shovalmishal/oriented-rcnn:v2
```
2. Then run the docker:
On the A5000:
```shell
docker run --gpus all -v $(pwd):/workspace/ -it shovalmishal/ad-stage1:v1


```
On runai:
```shell
runai submit --name ood-supervised-experiment2-test -g 1.0 -i shovalmishal/oriented-rcnn:v2 --pvc=storage:/storage --large-shm 
```

