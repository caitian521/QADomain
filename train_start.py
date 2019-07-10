#from main.main import start
#from net.qa_extract import QaExtract
from net.qa_extract import QaExtract
import torch
from util.Logginger import init_logger
from Io.data_loader import create_batch_iter
from train.train import fit
import config.args as args
from util.porgress_util import ProgressBar
from util.model_util import load_model
import os
from preprocessing.data_processor import read_squad_data
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def start():
    train_iter, num_train_steps = create_batch_iter("train")
    eval_iter = create_batch_iter("dev")

    epoch_size = num_train_steps * args.train_batch_size * args.gradient_accumulation_steps / args.num_train_epochs

    pbar = ProgressBar(epoch_size=epoch_size, batch_size=args.train_batch_size)

    #model = load_model(args.output_dir)
    model = QaExtract.from_pretrained(args.bert_model)  #QaExtract(args)
    '''
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    '''
        # ------------------判断CUDA模式----------------------
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()  # 多GPU
        # n_gpu = 1
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1

    logger = init_logger("torch", logging_path=args.log_path)
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    logger.info("epoch_size: {} num_train_steps: {}".format(epoch_size, num_train_steps))

    '''
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    '''
    #eval_acc, eval_f1, eval_loss_avg = evaluate(model, eval_iter, device)

    fit(model=model,
        training_iter=train_iter,
        eval_iter=eval_iter,
        num_epoch=args.num_train_epochs,
        pbar=pbar,
        num_train_steps=num_train_steps,
        device = device,
        n_gpu = n_gpu,
        verbose=1)


if __name__ == "__main__":
    #read_squad_data("data/big_train_data.json", "../data/")
    start()