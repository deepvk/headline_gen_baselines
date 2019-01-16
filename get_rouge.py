import sys
from rouge import Rouge

from utils import unbpe

with open(sys.argv[0]) as f:
    hyps = [line.strip() for line in f]
with open(sys.argv[1]) as f:
    refs = [line.strip() for line in f]

rouge = Rouge()
scores = rouge.get_scores([unbpe(hyp) for hyp in hyps], [unbpe(ref) for ref in refs])
print(scores)
