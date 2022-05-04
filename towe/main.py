import os

import fitlog

import argparse
import sys

import torch

from towe.tools.utils import MultiFocalLoss, tprint

from towe.model.ConfigParser import Config

from towe.model.trainer import Trainer, set_random_seed, load_data

from towe.model.Net import ExtractionNet, ExtractionNet_crf, ExtractionNet_mrc
from common import common_path

sys.path.append('../')


def my_bool(val):
    """

    :param val:
    :return:
    """
    if val == 'True':
        return True
    elif val == 'False':
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser = argparse.ArgumentParser()
parser.add_argument('--config_filename', type=str, default='conf_bert_gnn_lstm.ini')
parser.add_argument('--config_path', type=str, default=common_path.project_dir + '/towe/model/conf_bert_gnn_lstm.ini')
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--dataset_suffix', type=str, default='entire-space')
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--num_mid_layers', type=int, default=None)
parser.add_argument('--num_heads', type=int, default=None)
parser.add_argument('--threshold', type=int, default=None)
parser.add_argument('--train_batch_size', type=int, default=None)
parser.add_argument('--load_model_name', type=str, default='')
parser.add_argument('--save_model_name', type=str, default='')
parser.add_argument('--eval_frequency', type=int, default=1)
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--repeat', type=int, default=0)
parser.add_argument('--pretrained_bert_path', default=common_path.project_dir + '/models/bert-base-uncased', type=str)
parser.add_argument('--train_log', default=common_path.project_dir + '/log/train_log', type=str)
parser.add_argument('--val_log', default=common_path.project_dir + '/log/val_log', type=str)
args = parser.parse_args()

if __name__ == "__main__":
    configuration = args.__dict__
    args.config_path = os.path.join(common_path.project_dir + 'towe/model/config/', args.config_filename)

    save_model_name = 'config_filename_{config_filename}-dataset_{dataset}-repeat_{repeat}-' \
                      'entire_space_{dataset_suffix}.ckpt'.format_map(configuration)
    args.save_model_name = os.path.join(common_path.project_dir, 'models', save_model_name)

    # if configuration['entire_space']:
    #     args.data_path = os.path.join(common_path.data_base_dir, 'data-entire-space', args.dataset) + '/'
    # else:
    #     args.data_path = os.path.join(common_path.data_base_dir, 'data', args.dataset) + '/'
    args.data_path = os.path.join(common_path.data_base_dir, 'data-%s' % args.dataset_suffix, args.dataset) + '/'

    args.w2v_path = os.path.join(args.data_path, '../full_glove.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # fitlog.commit(__file__)    # 自动 commit 你的代码
    fitlog.set_log_dir(os.path.join(common_path.project_dir, 'logs/'))  # 设定日志存储的目录

    fitlog.add_hyper(args)  # 通过这种方式记录ArgumentParser的参数
    # fitlog.add_hyper_in_file(__file__)  #  记录本文件中写死的超参数

    config = Config(args.config_path)
    config.reset_config(args)
    config_dict = config.config_dicts
    default_config = config.config_dicts['default']
    preprocess_config = config.config_dicts['preprocess']
    model_config = config.config_dicts['model']

    print('default_config: %s' % str(default_config))
    print('preprocess_config: %s' % str(preprocess_config))
    print('model_config: %s' % str(model_config))

    num_class = 4

    set_random_seed(args.random_seed)

    loader = load_data(preprocess_config['data_path'],
                       preprocess_config,
                       model_config['train_batch_size'],
                       model_config['val_batch_size'],
                       default_config['use_bert'],
                       default_config['build_graph'])

    if default_config['use_bert']:
        word_embed_dim = 768
        word_emb_mode = "bert"
    else:
        word_embed_dim = 300
        word_emb_mode = "w2v"

    model_name = model_config['model']
    model = eval(model_name)(word_embed_dim=word_embed_dim,
                             output_size=num_class,
                             config_dicts=config_dict,
                             word_emb_mode=word_emb_mode,
                             graph_mode=default_config['build_graph'])

    print(model)

    config.print_config()

    assert model_config['loss'] in ["CrossEntropy", "FacalLoss"]
    if model_config['loss'] == "CrossEntropy":
        # loss_op = torch.nn.CrossEntropyLoss()
        loss_op = torch.nn.NLLLoss()
    else:
        loss_op = MultiFocalLoss(num_class=num_class, gamma=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

    trainer = Trainer(loader, model, loss_op, optimizer, args, config, fitlog_flag=True)
    trainer.load_model()
    trainer.train()

    fitlog.finish()  # finish the logging
