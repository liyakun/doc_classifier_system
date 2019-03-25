#!/usr/bin/env bash
wget ./data/ http://lampsrv02.umiacs.umd.edu/projdb/edit/userfiles/datasets/Tobacco3482_1.zip
wget ./data/ http://lampsrv02.umiacs.umd.edu/projdb/edit/userfiles/datasets/Tobacco3482_2.zip
mkdir ./data/imgs/
unzip ./data/Tobacco3482_1.zip ./data/imgs/
unzip ./data/Tobacco3482_2.zip ./data/imgs/
mkdir -p ./models/bvlc_alexnet/
wget ./models/bvlc_alexnet/bvlc_alexnet.caffemodel https://www.dropbox.com/s/sw37typiee0lags/bvlc_alexnet.caffemodel