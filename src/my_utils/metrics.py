import numpy as np
import collections
import re
import string
from collections import Counter

def normalize_answer(s):
    # s = 'answer:xxxxxx'
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def calculate_metric(pairs, metric_name):
    if metric_name == "accuracy": #for classification  one correct candidate
        return np.mean([pred == ans for (pred,ans) in pairs])
    elif metric_name == "f1":
        # For question answering   more than one candidates
        f1 = []
        for (pred,ans_s) in pairs:
            all_f1s = []
            if pred  == "CANNOTANSWER" or pred  == "no answer":
                f1.append(int(normalize_answer(ans_s[0]) == normalize_answer(pred)))
            else:
                for ans in ans_s:
                    ans = normalize_answer(ans)
                    pred = normalize_answer(pred)
                    if ans in pred:
                        all_f1s.append(1)
                    else:
                        prediction_tokens = pred.split()
                        ground_truth_tokens = ans.split()
                        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
                        num_same = sum(common.values())
                        if num_same == 0:
                            all_f1s.append(0)
                        else:
                            precision = 1.0 * num_same / len(prediction_tokens)
                            recall = 1.0 * num_same / len(ground_truth_tokens)
                            all_f1s.append((2 * precision * recall) / (precision + recall))
                f1.append(max(all_f1s))
        return np.mean(f1)