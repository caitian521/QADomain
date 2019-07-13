import time
import torch
from pytorch_pretrained_bert.optimization import BertAdam
from tensorboardX import SummaryWriter

import config.args as args
from util.plot_util import loss_acc_plot
from util.Logginger import init_logger
from evaluate.loss import loss_fn
from evaluate.acc_f1 import qa_evaluate
from util.model_util import save_model, load_model

logger = init_logger("torch", logging_path=args.log_path)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
import warnings
warnings.filterwarnings('ignore')


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

log_writer = SummaryWriter(args.log_file)
train_losses = []
train_accuracy = []

history = {
    "train_loss": train_losses,
    "train_acc": train_accuracy
}
def fit(model, training_iter, eval_iter, num_epoch, pbar, num_train_steps, device, n_gpu=1, verbose=1):

    # ---------------------优化器-------------------------
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    t_total = num_train_steps

    ## ---------------------GPU半精度fp16-----------------------------
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    ## ------------------------GPU单精度fp32---------------------------
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)
    # ---------------------模型初始化----------------------
    if args.fp16:
        model.half()

    model.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)

    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)  # , device_ids=[0, 1, 2]

    # 保存最好的模型
    best_f1 = 0
    start = time.time()
    global_step = 0
    for e in range(num_epoch):
        train_loss, train_acc, global_step = train_steps(model, training_iter, optimizer, pbar, global_step, t_total, start, device)
        eval_acc, eval_f1, eval_loss_avg = evaluate(model, eval_iter, device)

        if eval_f1 > best_f1:
            best_f1 = eval_f1
            save_model(model, args.output_dir, t_total*(e+1))

        logger.info(
            '\n\nEpoch %d - train_loss: %4f - eval_loss: %4f - train_acc:%4f - eval_acc:%4f - eval_f1:%4f\n'
            % (e + 1,
               train_loss.item(),
               eval_loss_avg,
               train_acc,
               eval_acc,
               eval_f1))
    loss_acc_plot(history)
# ------------------------训练------------------------------
def train_steps(model, training_iter, optimizer, pbar, global_step, t_total, start, device):
    model.train()
    save_step = t_total * args.eval_epoch
    max_f1 = 0
    for step, batch in enumerate(training_iter):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, start_positions, end_positions, answer_types, domain_types = batch
        start_logits, end_logits, answer_type_logits, domain_type_logits = model(input_ids, segment_ids, input_mask)
        train_loss = loss_fn(start_logits, end_logits, answer_type_logits, start_positions, end_positions, answer_types,
                             domain_type_logits, domain_types)

        if args.gradient_accumulation_steps > 1:
            train_loss = train_loss / args.gradient_accumulation_steps

        if args.fp16:
            optimizer.backward(train_loss)
        else:
            train_loss.backward()
            #train_loss.backward(torch.ones_like(train_loss))

        if (step + 1) % args.gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step += args.gradient_accumulation_steps


        start_logits, end_logits = start_logits.cpu(), end_logits.cpu()
        start_positions, end_positions = start_positions.cpu(), end_positions.cpu()
        train_acc, f1 = qa_evaluate((start_logits, end_logits), (start_positions, end_positions))

        #if e % verbose == 0:
        train_losses.append(train_loss.item())
        train_accuracy.append(train_acc)
        pbar.show_process(train_acc, train_loss.item(), f1, time.time() - start, step)

        if (global_step) % args.train_sample == 0:
            log_writer.add_scalar('Loss', train_loss.item(), global_step)
            log_writer.add_scalar('Acc', train_acc, global_step)

        '''
        模型选择
        '''
        if (step+1) % args.eval_step == 0 and step > save_step:
            if max_f1 < f1:
                max_f1 = f1
                save_model(model, args.output_dir, step)

    return train_loss, train_acc, global_step

    #loss_acc_plot(history)

# -----------------------验证----------------------------
def evaluate(model, eval_iter, device):
    model.eval()
    count = 0
    y_predicts, y_labels = [], []
    eval_starts_predict, eval_ends_predict = [], []
    eval_starts_label, eval_ends_label = [], []
    eval_type_label, eval_type_predict = [], []
    eval_loss, eval_acc, eval_f1 = 0, 0, 0
    with torch.no_grad():
        for step, batch in enumerate(eval_iter):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, start_positions, end_positions, answer_types, _ = batch
            start_logits, end_logits, answer_type_logits, _ = model(input_ids, segment_ids, input_mask)
            #eval_los = loss_fn(start_logits, end_logits, answer_type_logits, start_positions, end_positions, answer_types)
            #eval_loss = eval_los + eval_loss
            count += 1
            eval_starts_predict.append(start_logits)
            eval_ends_predict.append(end_logits)
            eval_starts_label.append(start_positions)
            eval_ends_label.append(end_positions)
            eval_type_predict.append(answer_type_logits)
            eval_type_label.append(answer_types)


        eval_starts_predicted = torch.cat(eval_starts_predict, dim=0).cpu()
        eval_ends_predicted = torch.cat(eval_ends_predict, dim=0).cpu()
        eval_starts_labeled = torch.cat(eval_starts_label, dim=0).cpu()
        eval_ends_labeled = torch.cat(eval_ends_label, dim=0).cpu()
        eval_type_predicted = torch.cat(eval_type_predict, dim=0).cpu()
        eval_type_labeled = torch.cat(eval_type_label, dim=0).cpu()

        eval_predicted = (eval_starts_predicted, eval_ends_predicted, eval_type_predicted)
        eval_labeled = (eval_starts_labeled, eval_ends_labeled, eval_type_labeled)

        eval_acc, eval_f1 = qa_evaluate(eval_predicted, eval_labeled)
        eval_loss_avg = 0
        return eval_acc, eval_f1, eval_loss_avg

