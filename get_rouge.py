import sys
from rouge import Rouge

from utils import unbpe

assert len(sys.argv) == 3, "Usage: get_rouge.py file_with_hypothesises files_with_references"

with open(sys.argv[1]) as f:
    hyps = [line.strip() for line in f]
with open(sys.argv[2]) as f:
    refs = [line.strip() for line in f]

rouge = Rouge()
hyps = [unbpe(hyp) for hyp in hyps]
refs = [unbpe(ref) for ref in refs]
scores = rouge.get_scores(hyps, refs, avg=True)
print(scores)
