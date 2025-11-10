python use_pytorch/server.py -nc 100 -cf 0.1 -E 1 -B 10 -mn mnist_cnn -ncomm 10 -iid 0 -lr 0.01 -vf 1 -g 0 -sf 5
# -nc 客户端数量
# -cf 通信比例
# -E 每次通信前每个客户端的训练轮数
# -B Batch Size
# -lr 学习率
# -ncomm 一共的通信次数
# -sf 存储频率