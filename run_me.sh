sed 's/nameserver 10.*/nameserver 132.66.150.2/' /etc/resolv.conf >/tmp/R$$
cp /tmp/R$$ /etc/resolv.conf

python ./tools/test.py ./configs/ood_experiments/test/3/oriented-rcnn-le90_r50_fpn_1x_dota.py /storage/shoval/OOD_experiments/3/train/epoch_12.pth --work-dir /storage/shoval/OOD_experiments/3/test
