import os

import argparse
import pandas as pd
import time

from nns.spv.base_mts_spv import BaseMTSSupervisor
from nns.spv.WDAN_spv import WDANSupervisor
from utils.config_utils import generate_spv_config
from utils.train_inits import init_seed

spv_map = {
    'iTransformer': BaseMTSSupervisor,
    'PatchTST': BaseMTSSupervisor,
    'Crossformer': BaseMTSSupervisor,
    'FEDformer': BaseMTSSupervisor,
    'WDAN': WDANSupervisor,
}

# region define parser
parser = argparse.ArgumentParser(description='WDAN')

# region general config
# --- basic config
parser.add_argument('--model', type=str, default='WDAN_iTransformer', help='models name')
parser.add_argument('--data', type=str, default='ETTh1', help='dataset name')
parser.add_argument('--fix_seed', type=int, default=2024, help='random seed')
parser.add_argument('--itr', type=int, default=1, help='number of iterations')
parser.add_argument('--machine', type=str, default='local', help='local')
parser.add_argument('--server_name', type=str, default='Local', help='server name')
parser.add_argument('--save_and_log', action='store_true', default=False, help='save and log')
parser.add_argument('--save_tb', action='store_true', default=False, help='save tensorboard')
parser.add_argument('--exp_id', type=str, default='', help='experiment id')

# --- data config
parser.add_argument('--seq_len', type=int, default=720, help='seq_len')
parser.add_argument('--pred_len', type=int, default=336, help='pred_len')
parser.add_argument('--label_len', type=int, default=360, help='pred_len')
parser.add_argument('--num_workers', type=int, default=8, help='num_workers for dataloader')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

# --- loss config
parser.add_argument('--loss_type', type=str, default='mse', help='loss type')

# --- train config
parser.add_argument('--gpu_id', type=int, default=0, help='id of gpu')
parser.add_argument('--precision', type=str, default='full', help='precision of training, [full, amp]')
parser.add_argument('--stats_strategy', type=str, default='union',
                    help='stats strategy, [union, stats_bb, stats_union, stats_bb_union]')
parser.add_argument('--base_lr', type=float, default=0.0001, help='base learning rate')
parser.add_argument('--base_stats_lr', type=float, default=0.0001, help='base stats learning rate')
parser.add_argument('--lr_decay_ratio', type=float, default=0.5, help='learning rate decay ratio')
parser.add_argument('--step_size', type=int, default=1, help='step size')
parser.add_argument('--early_stop', type=int, default=3, help='early stopping patience')
parser.add_argument('--epochs', type=int, default=10, help='epochs')
parser.add_argument('--stats_epochs', type=int, default=5, help='stats_epochs')
parser.add_argument('--twice_epoch', type=int, default=1,
                    help='twice epoch for three-step (like DDN) training')
# endregion

# region model / backbone Config
parser.add_argument('--use_norm', type=int, default=1, help='whether to use inst normalize; True 1 False 0')
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2024, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default=None,
                    help='down sampling method, only support avg, max, conv')

# PatchTST
parser.add_argument('--patch_len', type=int, default=16, help='patch length')

# endregion

# region stats config
parser.add_argument('--stats_wavelet', type=str, default='coif3', help='wavelet name')
parser.add_argument('--stats_dwt_levels', type=int, default=1, help='dwt levels')
parser.add_argument('--stats_filter_learn', action='store_false', default=False,
                    help='whether to learn filter')
parser.add_argument('--stats_d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--stats_d_ff', type=int, default=1024, help='dimension of fcn')
parser.add_argument('--stats_dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--stats_ffn_layers', type=int, default=2, help='num of ffn layers')
parser.add_argument('--stats_window_len', type=int, default=5, help='window len')

# endregion

# --- args parsing
args = parser.parse_args()


# endregion

# region results utils
def save_results_csv(val_loss, test_loss, run_id='', itr=0, file_path='results.csv'):
    records = vars(args)
    records.update({
        'itr': itr,
        'run_id': run_id,
        'runtime': time.strftime('%m%d%H%M'),
        'val_loss': val_loss,
        'test_mse': test_loss['mse_overall'],
        'test_mae': test_loss['mae_overall']
    })
    new_df = pd.DataFrame([records])
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        merged_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        merged_df = new_df

    # 保存数据（自动处理列对齐）
    merged_df.to_csv(file_path, index=False)
# endregion


def single_train():
    model_name = args.model
    data_name = args.data
    ret_save_dir = f'./rets'
    if not os.path.exists(ret_save_dir):
        os.makedirs(ret_save_dir)

    file_path = f'{ret_save_dir}/{args.exp_id}_{model_name}_{data_name}_{args.server_name}.csv' if args.exp_id else \
        f'{ret_save_dir}/{model_name}_{data_name}_{args.server_name}.csv'

    # training
    init_seed(args.fix_seed)
    if len(model_name.split('_')) > 1:
        stats_name, backbone_name = model_name.split('_')
        spv_generator = spv_map[stats_name]
    else:
        spv_generator = spv_map[model_name]
    for ii in range(args.itr):
        print(f">>>>>>Training {model_name} on {data_name} with itr {ii}")
        spv_config = generate_spv_config(args)
        spv = spv_generator(**spv_config)
        val_loss = spv.train()
        test_ret = spv.test(message=f's{args.fix_seed}_itr{ii}')['loss']

        save_results_csv(val_loss, test_ret,
                         run_id=spv.get_run_id(), itr=ii, file_path=file_path)


if __name__ == '__main__':
    single_train()
