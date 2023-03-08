import argparse
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
from utils.tools import dotdict



if __name__ == '__main__':
    args = dotdict()

    parser = argparse.ArgumentParser(description='Non-stationary Transformers for Time Series Forecasting')

    # basic config
    args.is_training=1
    args.model_id='test'
    args.model='Transformer'
    args.data='ETTh2'
    args.root_path='./data/ETT/'
    args.data_path='ETTh2.csv'
    args.features='M'
    args.target='OT'
    args.freq='h'

    args.checkpoints='./checkpoints/'

    # forecasting task
    args.seq_len=96
    args.label_len=48
    args.pred_len=96

    # model define
    args.enc_in=7
    args.dec_in=7
    args.c_out=7
    args.d_model=512
    args.n_heads=8
    args.e_layers=2
    args.d_layers=1
    args.d_ff=2048
    args.moving_avg=25
    args.factor=1
    args.distil='store_false'

    args.dropout=0.05
    args.embed='timeF'

    args.activation='activation'
    args.output_attention='store_true'
    args.do_predict='store_true'

    # optimization
    args.num_workers=10
    args.itr=2
    args.train_epochs=10
    args.batch_size=32
    args.patience=3
    args.learning_rate=0.0001
    args.des='test'
    args.loss='mse'
    args.lradj='type1'
    args.use_amp='store_true'

    args.use_gpu=True
    args.gpu=0
    args.use_multi_gpu='store_true'
    args.devices='0,1,2,3'
    args.seed=2021
    args.p_hidden_dims=[128, 128]
    args.p_hidden_layers=2
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.is_training = 1
    args.root_path = "./dataset/ETT-small/"
    args.data_path = "ETTh2.csv"
    args.model_id = "ETTh2_96_192"
    args.model = "Autoformer"
    args.data = "ETTh2"
    args.features = "M"
    args.seq_len = 96
    args.label_len = 48
    args.pred_len = 192
    args.e_layers = 2
    args.d_layers = 1
    args.enc_in = 7
    args.dec_in = 7
    args.c_out = 7
    args.gpu = 0
    args.des = 'Exp'
    args.itr = 1
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # if args.use_gpu:
    #     if args.use_multi_gpu:
    #         args.devices = args.devices.replace(' ', '')
    #         device_ids = args.devices.split(',')
    #         args.device_ids = [int(id_) for id_ in device_ids]
    #         args.gpu = args.device_ids[0]
    #     else:
    #         torch.cuda.set_device(args.gpu)

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    for ii in range(args.itr):  # itr就是实验次数可不是epoch，args.itr', type=int, default=2, help='experiments times')
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print(1)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)  # setting是用来保存模型的名字用的，很细节
        print(2)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
        print(3)

