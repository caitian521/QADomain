# ---------- Train -------------------
my_output = "output_domain_compress_sdn/"

log_path = my_output + "logs"
plot_path = my_output + "images/loss_acc.png"
data_dir = "data/"
output_dir = my_output + "checkpoint"
#output_dir = "output_sdn/checkpoint"
VOCAB_FILE = "pretrained_model/vocab.txt"
bert_model = "pretrained_model/pytorch_pretrained_model"
log_file = my_output + "logs"

import os
if not os.path.exists(my_output):
    os.makedirs(my_output)
    os.makedirs(log_path)
    os.makedirs(output_dir)
    os.makedirs(my_output+ "images")


#load_path = "output_domain_compress/checkpoint/pytorch_model0036150.bin"
load_path = "output_domain_compress_sdn/checkpoint/pytorch_model0036150.bin"


doc_stride = 128
max_query_length = 32
max_seq_length = 256
do_lower_case = True
train_batch_size = 36
eval_batch_size = 36
learning_rate = 2e-5
num_train_epochs = 6
warmup_proportion = 0.1
no_cuda = False
local_rank = -1
seed = 42
gradient_accumulation_steps = 2
fp16 = False
loss_scale = 0.

# -------------- train steps --------------------
train_sample = 100
eval_step = 200
eval_epoch = 2


answer_type = {"YES": 0, "NO": 1, "no-answer": 2, "long-answer": 3}
domain_type = {"civil": 0, "criminal": 1, "":3}

# ------------ Predict -----------------
predict_batch_size = 16
n_best_size = 1
max_answer_length = 256
verbose_logging = 1
version_2_with_negative = True
null_score_diff_threshold = 0
