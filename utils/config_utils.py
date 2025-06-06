from utils.path_utils import get_dir

data_path_config = {
    'Exchange': {
        'root_path': '/exchange_rate/',
        'data_name': 'Exchange',
        'data_path': 'exchange_rate.csv',
        'data': 'custom'
    },
    'ECL': {
        'root_path': '/electricity/',
        'data_name': 'ECL',
        'data_path': 'electricity.csv',
        'data': 'custom'
    },
    'ETTh1': {
        'root_path': '/ETT-small/',
        'data_name': 'ETTh1',
        'data_path': 'ETTh1.csv',
        'data': 'ETTh1'
    },
    'ETTh2': {
        'root_path': '/ETT-small/',
        'data_name': 'ETTh2',
        'data_path': 'ETTh2.csv',
        'data': 'ETTh2'
    },
    'ETTm1': {
        'root_path': '/ETT-small/',
        'data_name': 'ETTm1',
        'data_path': 'ETTm1.csv',
        'data': 'ETTm1'
    },
    'ETTm2': {
        'root_path': '/ETT-small/',
        'data_name': 'ETTm2',
        'data_path': 'ETTm2.csv',
        'data': 'ETTm2'
    },
    'Weather': {
        'root_path': '/weather/',
        'data_name': 'Weather',
        'data_path': 'weather.csv',
        'data': 'custom'
    },
}

model_params = {
    'iTransformer': [
        'seq_len',
        'pred_len',
        'use_norm',
        'd_model',
        'dropout',
        'factor',
        'n_heads',
        'e_layers',
        'd_ff',
        'activation'
    ],
    'PatchTST': [
        'seq_len',
        'pred_len',
        'use_norm',
        'patch_len',
        'd_model',
        'n_heads',
        'e_layers',
        'd_ff',
        'factor',
        'activation',
        'enc_in',
        'dropout'
    ],
    'Crossformer': [
        'seq_len',
        'pred_len',
        'enc_in',
        'e_layers',
        'd_model',
        'n_heads',
        'd_ff',
        'factor',
        'dropout'
    ],
    'FEDformer': [
        'seq_len',
        'label_len',
        'pred_len',
        'enc_in',
        'dec_in',
        'moving_avg',
        'd_model',
        'n_heads',
        'e_layers',
        'd_layers',
        'd_ff',
        'c_out',
        'embed',
        'freq',
        'dropout',
        'activation',
    ]
}

stats_params = {
    'WDAN': [
        'seq_len',
        'pred_len',
        'wavelet',
        'dwt_levels',
        'filter_learn',
        'window_len',
        'd_model',
        'd_ff',
        'dropout',
        'ffn_layers'
    ],
}


def generate_spv_config(args):
    args_dict = vars(args)
    current_dir_dict = get_dir(args.machine)
    base_data_dir = current_dir_dict['data_dir']
    base_model_dir = current_dir_dict['model_dir']

    model_name = args.model
    stats_flag = True if len(model_name.split('_')) == 2 else False
    data_name = args.data

    spv_config = {
        'pre_trained': False,
        'log_dir': f'{base_model_dir}/{model_name}',
        'log_level': 'INFO',
        'save_and_log': args.save_and_log,
        'save_tb': args.save_tb,
        'model_id': args.model
    }

    data_config = {
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'label_len': args.label_len,
        'num_workers': args.num_workers,
        'batch_size': args.batch_size,
        'embed': args.embed,
        'freq': args.freq,
        'features': 'M',
        'target': 'OT'
    }
    data_config.update(data_path_config[data_name])
    data_config['root_path'] = base_data_dir + data_config['root_path']

    if stats_flag:
        stats_direct_param = ['seq_len', 'pred_len', 'enc_in']
        stats_name, backbone_name = model_name.split('_')
        backbone_config = {param: args_dict[param] for param in model_params[backbone_name]}
        stats_config = {
            param: args_dict[param] if param in stats_direct_param else args_dict['stats_' + param]
            for param in stats_params[stats_name]
        }
        model_config = {
            'stats': stats_config,
            'backbone': backbone_config
        }
    else:
        model_config = {param: args_dict[param] for param in model_params[model_name]}

    loss_config = {'loss_type': args.loss_type}

    train_config = {
        'gpu_id': args.gpu_id,
        'precision': args.precision,
        'stats_strategy': args.stats_strategy,
        'base_lr': args.base_lr,
        'base_stats_lr': args.base_stats_lr,
        'lr_decay_ratio': args.lr_decay_ratio,
        'step_size': args.step_size,
        'early_stop': args.early_stop,
        'epochs': args.epochs,
        'stats_epochs': args.stats_epochs,
        'twice_epoch': args.twice_epoch,
        'debug': False,
        'cl_learn': False,
        'cl_step': 2000,
        'log_step': 200,
        'plot_loss': True,
        'optimizer': 'adam',
        'lr_type': 'StepLR',
        'weight_decay': 0.,
        'lr_milestones': [100],
        'clip_grad': False,
        'max_grad_norm': 5,
        'min_epochs': 0
    }

    spv_config['data'] = data_config
    spv_config['model'] = model_config
    spv_config['loss'] = loss_config
    spv_config['train'] = train_config

    return spv_config
