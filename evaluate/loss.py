from torch.nn import CrossEntropyLoss
import torch

def loss_fn(start_logits, end_logits, answer_type_logits, start_positions, end_positions, answer_types,
            domain_type_logits, domain_types):
    assert start_positions is not None and end_positions is not None
    # If we are on multi-GPU, split add a dimension
    if len(start_positions.size()) > 1:
        start_positions = start_positions.squeeze(-1)
    if len(end_positions.size()) > 1:
        end_positions = end_positions.squeeze(-1)
    # sometimes the start/end positions are outside our model inputs, we ignore these terms
    ignored_index = start_logits.size(1)
    start_positions.clamp_(0, ignored_index)
    end_positions.clamp_(0, ignored_index)
    loss_f = CrossEntropyLoss(ignore_index=ignored_index, reduce=False)

    start_loss = loss_f(start_logits, start_positions)
    end_loss = loss_f(end_logits, end_positions)

    pointer_loss = answer_type_logits[:,-1] * (start_loss + end_loss) + answer_type_logits[:,0] + answer_type_logits[:,1] +answer_type_logits[:,2]

    '''
    start_loss = loss_f(start_logits, start_positions)
    end_loss = loss_f(end_logits, end_positions)
    answer_type_loss = loss_f(answer_type_logits, answer_types)
    '''
    domain_type_loss = loss_f(domain_type_logits, domain_types)
    total_loss = (pointer_loss)*0.8 + domain_type_loss * 0.2

    return torch.mean(total_loss)