import os
import yaml

project_name = 'NsTimes'


def get_dir(machine='local'):
    if machine == 'local':
        return {
            'work_dir': './',
            'data_dir': './dataset',
            'model_dir': './models'
        }
    else:
        raise ValueError('machine name error')


def get_model_config_path(model_name, model_dir):
    return os.path.join(model_dir, f'{model_name}.yaml')


def load_spv_config(model_name='WDAN_iTransformer', dataset=None, machine='local'):
    current_dir_dict = get_dir(machine)
    work_dir = current_dir_dict['work_dir']
    base_data_dir = current_dir_dict['data_dir']
    base_model_dir = current_dir_dict['model_dir']

    model_dir = f'{work_dir}/data/model_configs' if dataset is None else f'{work_dir}/data/model_configs/{dataset}'
    with open(get_model_config_path(model_name, model_dir=model_dir)) as f:
        supervisor_config = yaml.safe_load(f)
    supervisor_config['log_dir'] = base_model_dir + supervisor_config['log_dir']
    supervisor_config['data']['root_path'] = base_data_dir + supervisor_config['data']['root_path']
    return supervisor_config


def get_work_dir(machine='local'):
    return get_dir(machine)['work_dir']
