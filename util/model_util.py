import os
import torch
from net.qa_extract import QaExtract
import config.args as args


def save_model(model, output_dir, step):
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    model_name = "pytorch_model%07d.bin" % step
    output_model_file = os.path.join(output_dir, model_name)
    torch.save(model_to_save.state_dict(), output_model_file)


def load_model(output_dir):
    # Load a trained model that you have fine-tuned
    output_model_file = os.path.join(output_dir)
    model_state_dict = torch.load(output_model_file)
    model = QaExtract.from_pretrained(args.bert_model, state_dict=model_state_dict)
    return model