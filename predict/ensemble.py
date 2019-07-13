import collections
import numpy as np
import pickle
import os
import json
import math

from pytorch_pretrained_bert.tokenization import BasicTokenizer, BertTokenizer
import config.args as args



def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    print("Writing predictions to: {}" .format(output_prediction_file))
    print("Writing nbest to: {}".format(output_nbest_file))

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

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            answer_type_indexs = _get_best_indexes(result.answer_type_logits, n_best_size)

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
            all_nbest_json[example.qas_id] = nbest_json


    with open(output_nbest_file, "w", encoding="utf-8") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
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

    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            print("Length not equal after stripping spaces: '{}' vs '{}'".format(orig_ns_text, tok_ns_text))
        return orig_text

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
            print("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            print("Couldn't map end position")
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



def ensemble_result(model_list, weight):
    result_list = [np.load(model) for model in model_list]
    model_num = len(model_list)
    result_num = len(result_list[0])

    unique_id_to_results = {}
    new_all_results = []
    RawResult = collections.namedtuple("RawResult",
                                       ["unique_id", "start_logits", "end_logits", "answer_type_logits"])

    for i in range(result_num):
        answer_type_logits = []
        start_logits = []
        end_logits = []
        unique_id = result_list[0][i].unique_id
        for j in range(model_num):
            unique_id_to_results.setdefault(result_list[i][0].unique_id, [  ]).append(result_list[i][j])

            answer_type_logits += np.array(result_list[i][j].answer_type_logits) * weight[j]
            start_logits += np.array(result_list[i][j].start_logits) * weight[j]
            end_logits += np.array(result_list[i][j].end_logits)  *weight[j]

        new_all_results.append(RawResult(unique_id=unique_id,
                                     start_logits=start_logits,
                                     end_logits=end_logits,
                                     answer_type_logits=answer_type_logits))
    return new_all_results, unique_id_to_results


if __name__ == "__main__":
    model_list = []
    weight = [1.0] * len(model_list)

    new_all_results, unique_id_to_results = ensemble_result(model_list, weight)

    example_path = ""
    feature_path = ""
    data_dir = "../result/"

    all_examples = pickle.loads(example_path)
    all_features = pickle.loads(feature_path)
    all_results = new_all_results
    n_best_size = args.n_best_size
    max_answer_length = args.max_answer_length
    do_lower_case = args.do_lower_case
    output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    output_nbest_file = os.path.join(data_dir, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")
    verbose_logging = args.verbose_logging
    version_2_with_negative = args.version_2_with_negative
    null_score_diff_threshold = args.null_score_diff_threshold

    write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold)

