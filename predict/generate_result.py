import torch
import os
import json
import math
import pickle
import collections
from tqdm import tqdm
from preprocessing.data_processor import read_squad_data, read_qa_examples, convert_examples_to_features
from net.qa_extract import QaExtract
from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer
import config.args as args
from util.Logginger import init_logger
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

logger = init_logger("bert_qa", logging_path=args.log_path)

device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")

def generate_result(model, tokenizer, test_raw_data, data_dir, model_name):
    read_squad_data(test_raw_data, data_dir, is_training=False)
    eval_examples = read_qa_examples(data_dir, corpus_type="test")
    eval_features = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        is_training=False)

    logger.info("***** Running predictions *****")
    logger.info("  Num orig examples = %d", len(eval_examples))
    logger.info("  Num split examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.predict_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)
    # Predicted by model. Note that one example turns into more than one features when processed by conver_example_to_features.py.
    # This means one example index corresponding to more than one features
    # In other word, for certain example, we can get more than one predict results by model.
    model.eval()
    all_results = []
    logger.info("Start evaluating")
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits", "answer_type_logits"])
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="Evaluating",
                                                                    disable=args.local_rank not in [-1, 0]):
        if len(all_results) % 1000 == 0:
            logger.info("Processing example: %d" % (len(all_results)))
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            batch_start_logits, batch_end_logits, batch_answer_type_logits, _ = model(input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            answer_type_logits = batch_answer_type_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(RawResult(unique_id=unique_id,
                                         start_logits=start_logits,
                                         end_logits=end_logits,
                                         answer_type_logits=answer_type_logits))

    result_path = '{}_result'.format(model_name)
    result_file = open(result_path, 'wb')
    pickle.dump(all_results, result_file)
    print('result {} saved!'.format(model))

    feature_path = '{}_features'.format(model_name)
    feature_file = open(feature_path, 'wb')
    pickle.dump(eval_features, feature_file)
    print('features {} saved!'.format(model))

    example_path = '{}_result'.format(model_name)
    example_file = open(example_path, 'wb')
    pickle.dump(eval_examples, example_file)
    print('examples {} saved!'.format(model))


def main(load_path, test_data_path, data_dir):
    '''
    :param load_path:  "output_sdn/checkpoint/bert_domain.bin"
    :param test_data_path:
    :param data_dir:
    :return:
    '''
    model_state_dict = torch.load(load_path)
    model = QaExtract.from_pretrained(args.bert_model, state_dict=model_state_dict)

    model.to(device)
    tokenizer = BertTokenizer(args.VOCAB_FILE)
    model_name = load_path.split("/")[-1].split(".")[0]
    generate_result(model, tokenizer, test_data_path, data_dir, model_name)

