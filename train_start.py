#from main.main import start
from net.qa_extract import QaExtract
from Io.data_loader import create_batch_iter
from train.train import fit
import config.args as args
from util.porgress_util import ProgressBar
from util.model_util import load_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

def start():
    train_iter, num_train_steps = create_batch_iter("train")
    eval_iter = create_batch_iter("dev")

    epoch_size = num_train_steps * args.train_batch_size * args.gradient_accumulation_steps / args.num_train_epochs

    pbar = ProgressBar(epoch_size=epoch_size, batch_size=args.train_batch_size)

    model = load_model(args.output_dir)
    #model = QaExtract.from_pretrained(args.bert_model)  #QaExtract(args)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    fit(model=model,
        training_iter=train_iter,
        eval_iter=eval_iter,
        num_epoch=args.num_train_epochs,
        pbar=pbar,
        num_train_steps=num_train_steps,
        verbose=1)


if __name__ == "__main__":
    start()