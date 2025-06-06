def get_loss_run_id(loss_configs, train_configs):
    loss_type = loss_configs.get('loss_type', 'mae')

    stats_strategy = train_configs.get('stats_strategy', 'stats_bb')
    stats_strategy = ''.join([_[0] for _ in stats_strategy.split('_')])
    learning_rate = train_configs.get('base_lr')
    stats_learning_rate = train_configs.get('base_stats_lr')
    step_size = train_configs.get('step_size', 1)
    lr_decay_ratio = train_configs.get('lr_decay_ratio', 0.1)
    early_stop = train_configs.get('early_stop', 3)
    epochs = train_configs.get('epochs', 10)
    twice_epoch = train_configs.get('twice_epoch', 1)

    loss_settings = f'_{loss_type}_{stats_strategy}_lr{learning_rate:.4f}_slr{stats_learning_rate:.4f}'
    loss_settings += f'_st{step_size}_ldr{lr_decay_ratio}_es{early_stop}_ep{epochs}_te{twice_epoch}'
    return loss_settings


def get_model_run_id(model_name, model_configs):
    if len(model_name.split('_')) == 2:
        stats_name, backbone_name = model_name.split('_')
        stats_run_id = get_model_run_id(stats_name, model_configs['stats'])
        backbone_run_id = get_model_run_id(backbone_name, model_configs['backbone'])
        return stats_run_id + backbone_run_id

    if model_name in ['WDAN']:
        wavelet = model_configs.get('wavelet', 'haar')
        dwt_levels = model_configs.get('dwt_levels', 1)
        filter_learn = 1 if model_configs.get('filter_learn', True) else 0
        window_len = model_configs.get('window_len', 5)
        d_model = model_configs.get('d_model', 128)
        d_ff = model_configs.get('d_ff', 256)
        dropout = model_configs.get('dropout', 0.1)
        ffn_layers = model_configs.get('ffn_layers', 1)
        model_settings = f'_{wavelet}_dl{dwt_levels}_fl{filter_learn}_wl{window_len}'
        model_settings += f'_dm{d_model}_df{d_ff}_do{dropout}_fl{ffn_layers}'
    elif model_name == 'iTransformer':
        e_layers = model_configs.get('e_layers', 1)
        d_model = model_configs.get('d_model', 128)
        d_ff = model_configs.get('d_ff', 256)
        use_norm = 1 if model_configs.get('use_norm', False) else 0
        model_settings = f'_el{e_layers}_dm{d_model}_df{d_ff}_norm{use_norm}'
    elif model_name == 'PatchTST':
        patch_len = model_configs.get('patch_len', 16)
        e_layers = model_configs.get('e_layers', 1)
        d_model = model_configs.get('d_model', 128)
        d_ff = model_configs.get('d_ff', 256)
        use_norm = 1 if model_configs.get('use_norm', False) else 0
        model_settings = f'_pl{patch_len}_el{e_layers}_dm{d_model}_df{d_ff}_norm{use_norm}'
    elif model_name == 'Crossformer':
        e_layers = model_configs.get('e_layers', 1)
        d_model = model_configs.get('d_model', 128)
        d_ff = model_configs.get('d_ff', 256)
        model_settings = f'_el{e_layers}_dm{d_model}_df{d_ff}'
    elif model_name == 'FEDformer':
        e_layers = model_configs.get('e_layers', 1)
        d_layers = model_configs.get('d_layers', 1)
        d_model = model_configs.get('d_model', 128)
        d_ff = model_configs.get('d_ff', 256)
        model_settings = f'_el{e_layers}_dl{d_layers}_dm{d_model}_df{d_ff}'
    else:
        raise ValueError(f'Invalid model name: {model_name}')

    return model_settings
        