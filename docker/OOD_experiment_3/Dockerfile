FROM shovalmishal/oriented-rcnn:base
COPY . /mmrotate
RUN pip install --no-cache-dir -e /mmrotate

RUN pip install ipdb
RUN pip install transformers=="4.29.2"
RUN pip install pytorch-lightning=="2.1.2"
RUN apt-get update && apt-get install nano
RUN pip install tensorboard=="1.15.0"
RUN pip install numpy=="1.22.0"
RUN pip install protobuf=="3.20.3"
WORKDIR /mmrotate
CMD ["/bin/bash", "-c", "chmod +x run_me.sh && ./run_me.sh"]
