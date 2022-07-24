import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'
w2v_path = os.path.join(project_dir, 'data/full_glove.txt')
bert_vocab_file_path = os.path.join(project_dir, 'data/vocab.txt')

data_base_dir = project_dir
