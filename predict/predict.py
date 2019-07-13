import torch
import os
import json
import math
import collections
from tqdm import tqdm
from preprocessing.data_processor import read_squad_data, read_qa_examples, convert_examples_to_features
from util.model_util import load_model
from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer
import config.args as args
from util.Logginger import init_logger
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

logger = init_logger("bert_qa", logging_path=args.log_path)

device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "answer_type_index", "start_logit", "end_logit",
         "start_cls_logit", "end_cls_logit", "answer_type_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        # score_null = 1000000  # large and positive
        # min_null_feature_index = 0  # the paragraph slice with min null score
        # null_start_logit = 0  # the start logit at the slice with min null score
        # null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            answer_type_indexs = _get_best_indexes(result.answer_type_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            # if version_2_with_negative:
            #     feature_null_score = result.start_logits[0] + result.end_logits[0]
            #     if feature_null_score < score_null:
            #         score_null = feature_null_score
            #         min_null_feature_index = feature_index
            #         null_start_logit = result.start_logits[0]
            #         null_end_logit = result.end_logits[0]
            for start_index, answer_type_index in zip(start_indexes, answer_type_indexs):
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            answer_type_index=answer_type_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                            start_cls_logit = result.start_logits[0],
                            end_cls_logit = result.end_logits[0],
                            answer_type_logit=result.answer_type_logits[answer_type_index]))
        # if version_2_with_negative:
        #     prelim_predictions.append(
        #         _PrelimPrediction(
        #             feature_index=min_null_feature_index,
        #             start_index=0,
        #             end_index=0,
        #             start_logit=null_start_logit,
        #             end_logit=null_end_logit,
        #         ))
        ## solution 1
        # prelim_predictions = sorted(
        #     prelim_predictions,
        #     key=lambda x: (x.start_logit + x.end_logit - x.start_cls_logits - x.end_cls_logits),
        #     reverse=True)
        ## solution 2
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: x.answer_type_logit,
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "answer_type_index"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0 and pred.answer_type_index == args.answer_type["long-answer"]:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                if pred.answer_type_index == args.answer_type['no-answer']:
                    final_text = ""
                else:
                    if pred.answer_type_index == args.answer_type["YES"]:
                        final_text = "YES"
                    else:
                        final_text = "NO"
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    answer_type_index=pred.answer_type_index))
        # # if we didn't include the empty option in the n-best, include it
        # if version_2_with_negative:
        #     if "" not in seen_predictions:
        #         nbest.append(
        #             _NbestPrediction(
        #                 text="",
        #                 start_logit=null_start_logit,
        #                 end_logit=null_end_logit))

        #     # In very rare edge cases we could only have single null prediction.
        #     # So we just create a nonce prediction in this case to avoid failure.
        #     if len(nbest) == 1:
        #         nbest.insert(0,
        #                      _NbestPrediction(text="", start_logit=0.0, end_logit=0.0))
        #
        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="", answer_type_index=args.answer_type["no-answer"]))

        assert len(nbest) >= 1

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["answer_type"] = entry.answer_type_index
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            # score_diff = score_null - best_non_null_entry.start_logit - (
            #     best_non_null_entry.end_logit)
            # scores_diff_json[example.qas_id] = score_diff
            # if score_diff > null_score_diff_threshold:
            #     all_predictions[example.qas_id] = ""
            # else:
            #     all_predictions[example.qas_id] = best_non_null_entry.text
            all_nbest_json[example.qas_id] = nbest_json

    #with open(output_prediction_file, "w") as writer:
    #    writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n")

    with open(output_nbest_file, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

   # if version_2_with_negative:
   #     with open(output_null_log_odds_file, "w") as writer:
   #         writer.write(json.dumps(scores_diff_json, indent=4, ensure_ascii=False) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
       # if verbose_logging:
       #     logger.info(
       #        "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def make_predict(model, tokenizer, test_raw_data, data_dir):
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

    output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    output_nbest_file = os.path.join(data_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")
    # post process
    write_predictions(eval_examples, eval_features, all_results,
                      args.n_best_size, args.max_answer_length,
                      args.do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                      args.version_2_with_negative, args.null_score_diff_threshold)


def main(test_data_path, data_dir):
    model = load_model(args.load_path)
    model.to(device)
    tokenizer = BertTokenizer(args.VOCAB_FILE)
    make_predict(model, tokenizer, test_data_path, data_dir)

