# Docker instructions
1. Build the docker:
```shell
docker build -t oriented-rcnn:base -f docker/OOD_base_experiment/Dockerfile .
docker tag oriented-rcnn:base shovalmishal/oriented-rcnn:base
docker push shovalmishal/oriented-rcnn:base
```
