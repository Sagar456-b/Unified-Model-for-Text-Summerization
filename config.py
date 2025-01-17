import  torch
from typing import Optional

input_unit = 128
dec_unit = 128
embed_size = 256
pointer = True

# Data
# # max_vocab_size = 20000
# embed_file = None  # use pre-trained embeddings
source = 'big_samples'    # use value: train or  big_samples
# data_path: str = '../files/train.txt'
# val_data_path = '../files/dev.txt'
# test_data_path = '../files/test.txt'
# stop_word_file: str = '../files/HIT_stop_words.txt'
# max_src_len = 300  # exclusive of special tokens such as EOS
# max_tgt_len = 100  # exclusive of special tokens such as EOS
truncate_src = True
truncate_tgt = True
min_dec_steps = 30
max_dec_steps = 100
enc_rnn_dropout = 0.5
enc_attn = True
dec_attn = True
dec_in_dropout = 0
dec_rnn_dropout = 0
dec_out_dropout = 0


# Training
trunc_norm_init_std = 1e-4
eps = 1e-31
learning_rate = 0.001
lr_decay = 0.0
initial_accumulator_value = 0.1
epochs = 1
batch_size = 32
coverage = True
fine_tune = False
scheduled_sampling = False
weight_tying = False
max_grad_norm = 2.0
is_cuda = True
DEVICE = torch.device("cuda" if is_cuda else "cpu")
LAMBDA = 1




if pointer:
    if coverage:
        if fine_tune:
            model_name = 'ft_pgn'
        else:
            model_name = 'cov_pgn'
    elif scheduled_sampling:
        model_name = 'ss_pgn'
    elif weight_tying:
        model_name = 'wt_pgn'
    else:
        if source == 'big_samples':
            model_name = 'pgn_big_samples'
        else:
            model_name = 'pgn'
else:
    model_name = 'baseline'