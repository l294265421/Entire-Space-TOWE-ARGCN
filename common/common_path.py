import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
bert_vocab_file_path = '/Users/yuncongli/PycharmProjects/data/uncased_L-12_H-768_A-12/vocab.txt'
w2v_path = os.path.join(project_dir, 'data/full_glove.txt')

if sys.platform != 'win32' and sys.platform != 'darwin':
    project_dir = '/data/ceph/yuncongli/towe-eacl-private/'
    bert_vocab_file_path = '/data/ceph/yuncongli/word-vector/uncased_L-12_H-768_A-12/vocab.txt'
    w2v_path = os.path.join(project_dir, 'data/full_glove.txt')

data_base_dir = project_dir
