PYTORCH=/root/anaconda3/envs/CenterNet/lib/python3.6/site-packages/
 # usually ~/anaconda3/envs/CenterNet/lib/python3.6/site-packages/
# for pytorch v0.4.0
#sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
# for pytorch v0.4.1
sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py

