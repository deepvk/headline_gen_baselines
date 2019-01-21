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

print("rouge-1-f\trouge-1-p\trouge-1-r\trouge-2-f\trouge-2-p\trouge-2-r\trouge-l-f\trouge-l-p\trouge-l-r")
print("\t".join([str(scores["rouge-1"]["f"])[:10], str(scores["rouge-1"]["p"])[:10], str(scores["rouge-1"]["r"])[:10],
                 str(scores["rouge-2"]["f"])[:10], str(scores["rouge-2"]["p"])[:10], str(scores["rouge-2"]["r"])[:10],
                 str(scores["rouge-l"]["f"])[:10], str(scores["rouge-l"]["p"])[:10], str(scores["rouge-l"]["r"])[:10]]))
