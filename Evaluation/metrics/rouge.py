#Calculate F-1 of lexical overlap by LCS: longest common subsequence (in terms of word order, adjacency not required)
#Precision = len(LCS) / len(Answer)
#Recall = len(LCS) / len(Ref)
from rouge_score import rouge_scorer
def compute_rouge(
    answer: str,
    ground_truth: str,
    rouge_type: str = "rougeL",
    mode: str = "fmeasure"
    ) -> float:
    """
    Compute ROUGE score between generated answer and ground truth reference.
    """
    if not ground_truth.strip() or not answer.strip():
        return 0.0
    
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    scores = scorer.score(ground_truth, answer)
    return getattr(scores[rouge_type], mode)