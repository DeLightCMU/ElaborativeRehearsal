

def eval_precisions(scores, targets, topk=(1,)):
  """Computes the precision@k for the specified values of k
  Args:
    scores: FloatTensor, (num_examples, num_classes)
    targets: LongTensor, (num_examples, )
  """
  maxk = max(topk)
  num_examples = targets.size(0)

  _, preds = scores.topk(maxk, 1, largest=True, sorted=True)
  preds = preds.t() # (maxk, num_exmples)
  correct = preds.eq(targets.unsqueeze(0).expand_as(preds))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0 / num_examples).data.item())
  return res

