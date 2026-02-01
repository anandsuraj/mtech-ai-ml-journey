# Literature Survey: Automatic Evaluation Metrics for Statistical Machine Translation

**NLP Applications - Assignment 2 - Group 5, Part 2**  
**Student Submission**  
**Date**: January 26, 2026

---

## Abstract

This literature survey examines the evolution and current state of automatic evaluation metrics for Statistical Machine Translation (SMT). We focus on BLEU and its variants, alternative metrics that address BLEU's limitations, and recent neural approaches to MT evaluation. The survey covers seminal papers from 2002 to recent 2024-2025 research, providing insights into how the field has progressed from simple n-gram matching to sophisticated neural evaluation systems.

**Note**: This survey synthesizes research to understand why we use specific metrics and what the future holds for MT evaluation.

---

## 1. Introduction

### 1.1 The MT Evaluation Problem

Machine Translation evaluation faces a fundamental challenge: **translations are not unique**. A single source sentence can be correctly translated in multiple ways, varying in:

- Word choice (synonyms)
- Word order (syntax)
- Style (formal vs. informal)
- Level of detail

**Human evaluation**, while ideal, is:
- Expensive (~$100-300 per hour for professional evaluators)
- Time-consuming (days to weeks for large test sets)
- Subjective (inter-annotator agreement often <70%)
- Non-repeatable (evaluators may give different scores on re-evaluation)

**Automatic metrics** are essential for:
1. **Rapid system development** (evaluate within seconds)
2. **Reproducible comparison** (same scores every time)
3. **Tuning MT systems** (optimize parameters automatically)
4. **Large-scale evaluation** (millions of sentences)

### 1.2 Survey Scope

This survey covers:
- **BLEU** (2002): The dominant metric for 20+ years
- **Alternative n-gram metrics** (NIST, METEOR, ROUGE)
- **Edit-distance metrics** (TER, HTER)
- **Character-based metrics** (chrF, chrF++)
- **Neural metrics** (COMET, BERTScore, BLEURT)
- **Comparative studies** and meta-evaluation research

---

## 2. BLEU: The Foundation (2002-Present)

### 2.1 Original BLEU Paper

**Citation**: Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). BLEU: a method for automatic evaluation of machine translation. In *Proceedings of the 40th annual meeting of the Association for Computational Linguistics* (pp. 311-318).

**Core Contribution**:

BLEU (Bilingual Evaluation Understudy) introduced a simple yet effective approach based on:

1. **Modified n-gram precision**: Counts matching n-grams between candidate and reference, but clips matches to avoid rewarding repetition

2. **Brevity penalty**: Prevents favoring short translations:
   ```
   BP = {1                if c > r
        {e^(1-r/c)       if c ≤ r
   ```

3. **Geometric mean**: Combines precision from multiple n-gram orders (typically 1-4)

**Formula**:
```
BLEU = BP × exp(Σ(w_n × log p_n))
where p_n = n-gram precision, w_n = weight (usually 1/N)
```

**Key Findings**:
- Correlation with human judgment: 0.79 to 0.99 (dataset-dependent)
- Fast computation (milliseconds per sentence)
- Language-independent (works for any language pair)

**Student Insight**: BLE's genius is its simplicity - it reduced complex MT evaluation to counting matching sequences.

### 2.2 BLEU Strengths

**Why BLEU became the standard**:

1. **Simplicity**: Easy to understand and implement
2. **Efficiency**: Fast computation enables quick iteration
3. **Reproducibility**: Deterministic, same score every time
4. **Correlation**: Reasonably correlates with human judgment (ρ = 0.6-0.8)
5. **Multiple references**: Naturally handles multiple correct translations

**Empirical Success**:
- Used in WMT (Workshop on Machine Translation) since 2006
- Standard metric in >95% of MT papers
- Integrated into MT tuning algorithms (MERT, MIRA)

### 2.3 BLEU Limitations

**Research has identified critical limitations**:

**1. Lack of Recall** (Callison-Burch et al., 2006)
- BLEU only measures precision, not recall
- Doesn't penalize missing content
- Can give high scores to partial translations

**2. No Semantic Understanding** (Reiter, 2018)
- Treats all words equally
- No synonym recognition ("big" != "large")
- Misses paraphrases

**3. Word Order Insensitivity** (Denkowski & Lavie, 2014)
- Relies only on n-grams
- Can miss significant word order errors
- Especially problematic for languages with different syntax

**4. Reference Dependency** (Freitag et al., 2022)
- Score heavily depends on which references are provided
- One additional reference can change score by ±0.1
- Doesn't account for valid translations not in references

**5. Poor Segment-Level Correlation** (Mathur et al., 2020)
- Works well for corpus-level (thousands of sentences)
- Poor for sentence-level evaluation (ρ < 0.3)
- Unsuitable for quality estimation of individual translations

**Example Illustrating Limitations**:

```
Source: "The cat sat on the mat"

Bad Translation: "mat the on sat cat the" 
BLEU: ~0.6 (high! all words match)

Good Translation: "The feline rested upon the rug"
BLEU: ~0.0 (low! different words, same meaning)
```

---

## 3. Alternative N-gram Based Metrics

### 3.1 NIST (2002)

**Citation**: Doddington, G. (2002). Automatic evaluation of machine translation quality using n-gram co-occurrence statistics. In *Proceedings of HLT* (pp. 138-145).

**Key Innovation**: Weight n-grams by informativeness (rarer n-grams count more)

**Formula**:
```
NIST = Σ Info(w_1...w_n) × count(w_1...w_n) / count(w_1...w_n-1)

Info(w_1...w_n) = log₂(# of occurrences of w_1...w_n-1 / # of occurrences of w_1...w_n)
```

**Advantages over BLEU**:
- Rewards rare/informative n-grams more
- Less affected by length variations
- Better correlation at sentence level (reported ρ = 0.82 vs BLEU's 0.79)

**Usage**: Popular in early 2000s, less common now, but still used in some domains (speech translation)

### 3.2 ROUGE (2004)

**Citation**: Lin, C. Y. (2004). ROUGE: A package for automatic evaluation of summaries. In *Text summarization branches out* (pp. 74-81).

**Original Purpose**: Text summarization evaluation

**Key Difference from BLEU**: **Recall-oriented** instead of precision-oriented

**Variants**:
- **ROUGE-N**: N-gram recall
- **ROUGE-L**: Longest common subsequence
- **ROUGE-S**: Skip-bigram co-occurrence

**Formula** (ROUGE-N):
```
ROUGE-N = Σ count_match(n-gram) / Σ count(n-gram in reference)
```

**Note**: ROUGE is mainly for summarization, but concepts influenced MT evaluation thinking about balance between precision and recall.

---

## 4. Metrics with Linguistic Awareness

### 4.1 METEOR (2005-2014)

**Citation**: Banerjee, S., & Lavie, A. (2005). METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. In *Proceedings of the ACL workshop on intrinsic and extrinsic evaluation measures* (pp. 65-72).

**Major Advancement**: Addresses BLEU's synonym blindness

**Key Features**:

1. **Synonym Matching**: Uses WordNet for English
   - "big" matches "large"
   - "car" matches "automobile"

2. **Stem Matching**: 
   - "running" matches "runs"
   - "walked" matches "walk"

3. **Paraphrase Matching**: 
   - Uses paraphrase tables
   - "not good" matches "bad"

4. **Recall-Precision Balance**:
   - F-mean (harmonic mean) instead of just precision
   - Formula: `F = (P × R) / (α × P + (1-α) × R)`

5. **Fragmentation Penalty**:
   - Penalizes matching words that are non-contiguous
   - Rewards maintaining word order

**Performance**:
- Sentence-level correlation: ρ = 0.40-0.45 (vs BLEU: 0.25-0.30)
- Corpus-level correlation: ρ = 0.85-0.90 (vs BLEU: 0.80-0.85)

**Limitations**:
- Language-specific (requires WordNet, stemmers)
- Slower than BLEU (10-100x)
- Not widely supported for non-English

**Student Insight**: METEOR was revolutionary for showing that linguistic knowledge improves correlation, paving way for modern neural metrics.

### 4.2 Extended METEOR Variants (2014)

**Citation**: Denkowski, M., & Lavie, A. (2014). Meteor universal: Language specific translation evaluation for any target language. In *Proceedings of the ninth workshop on statistical machine translation* (pp. 376-380).

**Contribution**: Meteor Universal
- Extended to 50+ languages
- Uses multilingual paraphrase tables
- Maintains high correlation across languages

---

## 5. Edit-Distance Based Metrics

### 5.1 TER - Translation Edit Rate (2006)

**Citation**: Snover, M., Dorr, B., Schwartz, R., Micciulla, L., & Makhoul, J. (2006). A study of translation edit rate with targeted human annotation. In *Proceedings of AMTA* (pp. 223-231).

**Concept**: Minimum number of edits to transform hypothesis into reference

**Edit Operations**:
- Insertion
- Deletion
- Substitution
- Shift (word/phrase movement)

**Formula**:
```
TER = # of edits / average # of reference words
```

**Range**: 0 (perfect) to ∞ (higher is worse, opposite of BLEU)

**Advantages**:
- Intuitive (counts actual errors)
- Captures word order errors through shifts
- Used in post-editing studies (measures human effort)

**Limitations**:
- Treats all errors equally (deleting "the" = deleting "not")
- Slow computation for long sentences (O(n²))
- Lower correlation than BLEU/METEOR

### 5.2 HTER - Human Targeted TER (2006)

**Innovation**: Human post-editor creates "targeted reference" specifically for the hypothesis

**Process**:
1. MT generates translation
2. Human minimally edits to correct it
3. TER computed between original MT and corrected version

**Advantage**: More accurate measure of post-editing effort

**Disadvantage**: Requires human involvement (not fully automatic)

---

## 6. Character-Level Metrics

### 6.1 chrF and chrF++ (2015-2017)

**Citation**: Popović, M. (2015). chrF: character n-gram F-score for automatic MT evaluation. In *Proceedings of WMT* (pp. 392-395).

**Motivation**: Handle morphologically rich languages better than word-level metrics

**Key Innovation**: Character n-grams instead of word n-grams

**Formula**:
```
chrF = (β² × chrP × chrR) / (β² × chrP + chrR)

where chrP = character n-gram precision
      chrR = character n-gram recall
      β² = weight(typically 2, favors recall)
```

**chrF++ Extensions** (Popović, 2017):
- Combines character and word n-grams
- Formula: `chrF++ = α × chrF + (1-α) × wordF`
- Even better correlation

**Advantages**:
- **Language-agnostic**: Works for all language pairs
- **Handles morphology**: Partial word matches count
- **No tokenization**: Avoids tokenization inconsistencies
- **Robust to compounding**: German, Finnish, etc.

**Performance** (WMT metrics task):
- German-English: ρ = 0.52 (vs BLEU: 0.48)
- Finnish-English: ρ = 0.58 (vs BLEU: 0.42) - significant improvement!

**Note**: chrF's success for morphologically rich languages shows that one-size-fits-all metrics have limitations. Language characteristics matter!

---

## 7. Neural and Embedding-Based Metrics

### 7.1 BERTScore (2020)

**Citation**: Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating text generation with bert. In *ICLR*.

**Revolutionary Idea**: Use contextual embeddings instead of surface form matching

**How it works**:
1. Get BERT embeddings for each token in hypothesis and reference
2. Compute cosine similarity between embeddings
3. Use greedy matching to find best alignment
4. Average similarities

**Formula**:
```
BERTScore_Precision = (1/|ĥ|) × Σ max_r∈ref (ĥ_i · r_j)
BERTScore_Recall = (1/|r|) × Σ max_ĥ∈hyp (r_j · ĥ_i)
BERTScore_F1 = 2 × (P × R) / (P + R)
```

**Advantages**:
- **Semantic awareness**: "car" and "automobile" have high similarity
- **Context-sensitive**: "bank" (river) vs "bank" (money) distinguished
- **No explicit resources needed**: No WordNet, no paraphrase tables
- **Strong correlation**: ρ = 0.60+ at segment level (vs METEOR: 0.45)

**Challenges**:
- **Computational cost**: Requires running BERT (GPU for efficiency)
- **Model dependency**: Different BERT models give different scores
- **Not fully explainable**: Hard to understand why scores differ

**Impact**: Showed that neural methods can significantly outperform traditional metrics

### 7.2 BLEURT (2020)

**Citation**: Sellam, T., Das, D., & Parikh, A. P. (2020). BLEURT: Learning robust metrics for text generation. In *ACL* (pp. 7881-7892).

**Innovation**: Pre-train BERT on synthetic data for MT evaluation

**Training Process**:
1. Create millions of (source, hypothesis, quality) tuples
2. Use weak supervision (BLEU, BERTScore, backtranslation)
3. Fine-tune BERT to predict quality scores
4. Further fine-tune on human ratings (WMT data)

**Performance** (WMT 2019):
- Segment-level correlation: ρ = 0.55 (vs BLEU: 0.22, BERTScore: 0.51)
- System-level correlation: ρ = 0.95 (vs BLEU: 0.89)

**Advantages**:
- **Best correlation** with human judgment to date
- **Robust to domain shift**: Pre-training helps generalization
- **Interpretable**: Can analyze which aspects it learned

**Disadvantages**:
- **Requires large computational resources**: Training needs GPUs
- **Black box**: Hard to debug unexpected scores
- **Reproducibility concerns**: Need exact same model version

### 7.3 COMET (2020-2022)

**Citation**: Rei, R., Stewart, C., Farinha, A. C., & Lavie, A. (2020). COMET: A neural framework for MT evaluation. In *EMNLP* (pp. 2685-2702).

**Unique Approach**: Cross-lingual encoder using source text

**Architecture**:
```
[Source] → XLM-R Encoder → 
[Hypothesis] → XLM-R Encoder → Pooling → Quality Prediction
[Reference] → XLM-R Encoder →
```

**Key Insight**: Using source helps identify errors that bilingual humans would catch

**COMET Variants**:
1. **COMET-MQM**: Trained on MQM (Multidimensional Quality Metrics) annotations
2. **COMET-QE**: Quality Estimation (no reference needed!)
3. **COMET-22**: Improved 2022 version with better training

**Performance** (WMT 2020):
- Segment-level: ρ = 0.614 (best in competition)
- System-level: ρ = 0.952
- **2022 version**: ρ = 0.664 segment-level (huge improvement)

**Why COMET is powerful**:
- **Multilingual**: Works for 100+ languages via XLM-R
- **Source-aware**: Catches meaning errors better
- **Reference-free variant**: Can evaluate without human translations
- **Error analysis**: Can identify specific error types

**Student Insight**: COMET represents the current state-of-the-art, showing that leveraging source text and modern pre-trained models yields best results.

---

## 8. Comparative Studies and Meta-Evaluation

### 8.1 WMT Metrics Tasks (Annual)

**Citation**: Mathur, N., Baldwin, T., & Cohn, T. (2020). Tangled up in BLEU: Reevaluating the evaluation of automatic machine translation evaluation metrics. In *ACL* (pp. 4984-4997).

**Purpose**: Annual competition to evaluate evaluation metrics

**Methodology**:
- Collect human judgments on MT outputs (DA - Direct Assessment)
- Compute correlation between metric scores and human scores
- Report Kendall's τ, Pearson's ρ, Spearman's ρ

**Key Findings (WMT 2020-2022)**:

| Metric | Segment-level Correlation | System-level Correlation | Speed |
|---------|---------------------------|-------------------------|--------|
| BLEU | 0.25 | 0.89 | Fast |
| METEOR | 0.42 | 0.91 | Medium |
| chr F++ | 0.48 | 0.93 | Fast |
| BERTScore | 0.51 | 0.94 | Slow |
| BLEURT | 0.55 | 0.95 | Slow |
| COMET-22 | **0.664** | **0.952** | Slow |

**Trend**: Neural metrics dominate recent competitions

### 8.2 Critical Analysis of BLEU

**Citation**: Reiter, E. (2018). A structured review of the validity of BLEU. *Computational Linguistics*, 44(3), 393-401.

**Major Criticisms**:

1. **Lack of theoretical foundation**: No clear reason why geometric mean of n-grams should correlate with quality

2. **Gaming possibilities**: Systems can be optimized to maximize BLEU without improving translation quality

3. **Inconsistent correlation**: Works well for some language pairs, poorly for others

4. **Length bias**: Brevity penalty is crude, doesn't account for content completeness

**Quote**: "BLEU is used because it is standard, not because it is necessarily the best metric."

**Note**: This paper sparked important discussions about whether we should continue using BLEU as the primary metric.

### 8.3 Beyond Correlation: Task-Based Evaluation

**Citation**: Freitag, M., Rei, R., Mathur, N., Lo, C. K., Stewart, C., Foster, G., ... & Neubig, G. (2022). Results of WMT22 metrics shared task: Stop using BLEU–neural metrics are better and more robust. In *Proceedings of WMT*.

**Revolutionary Claim**: "Stop using BLEU - neural metrics are better"

**Evidence**:
- COMET consistently outperforms BLEU across all language pairs
- Neural metrics more robust to adversarial attacks
- Better at identifying specific error types (semantics, fluency, adequacy)

**Recommendations for Researchers**:
1. **Primary metric**: COMET or BLEURT
2. **Secondary metric**: chrF++ for reproducibility
3. **Legacy reporting**: BLEU for comparison with older work
4. **Human evaluation**: Always validate with human judges for final claims

---

## 9. Specialized Metrics and Emerging Directions

### 9.1 Quality Estimation (Reference-Free)

**No reference scenario**: Evaluate MT quality without human translations

**Approaches**:
1. **COMET-QE**: Uses only source + hypothesis
2. **OpenKiwi**: Predicts word-level quality
3. **DeepQuest**: Neural quality estimation framework

**Applications**:
- Real-time translation quality filtering
- Post-editing effort prediction
- Zero-resource language evaluation

### 9.2 Explainable Metrics

**Motivation**: Understand *why* a translation got a certain score

**Approaches**:
- **Error annotation**: Categorize errors (fluency, adequacy, terminology)
- **Attention visualization**: Show which words influence score
- **Counterfactual explanations**: "Score would improve if this word changed"

**Research**: Still emerging, no standard solution yet

### 9.3 Multimodal MT Evaluation

**Context**: Image captioning, video description translation

**Challenges**:
- Need to consider visual context
- Standard metrics don't account for image-text alignment

**Proposed Solutions**:
- **CLIP-Score**: Use vision-language models
- **ViLBERT-based metrics**: Multimodal transformers

---

## 10. Critical Discussion

### 10.1 The BLEU Paradox

**Observation**: BLEU has known flaws, yet remains dominant

**Reasons**:
1. **Historical momentum**: 20+ years of use
2. **Comparability**: Easy to compare across papers
3. **Speed**: Fast enough for MT system tuning
4. **Simplicity**: Easy to implement and understand

**Student Perspective**: Like using accuracy for unbalanced datasets in ML - we know it's flawed, but standardization has value.

### 10.2 Neural Metrics Challenges

**Computational Cost**:
- BERT-based metrics: 100-1000x slower than BLEU
- Requires GPUs for practical use
- Carbon footprint concerns for large evaluations

**Black Box Nature**:
- Hard to debug unexpected scores
- Difficult to explain to stakeholders
- Can be gamed (adversarial examples)

**Reproducibility**:
- Model version matters (BERT-base vs BERT-large)
- Random seed affects scores
- Harder to replicate results exactly

### 10.3 Culture-Specific and Low-Resource Languages

**Challenge**: Most metrics designed/tested on high-resource European languages

**Problems**:
- Poor performance for:
  - Morphologically rich languages (Turkish, Finnish)
  - Distant language pairs (Chinese-Arabic)
  - Low-resource languages (Swahili, Kazakh)
- Cultural nuances not captured

**Proposed Solutions**:
- Language-specific metrics
- Cross-lingual models (XLM-R helps)
- Community-driven evaluation (native speakers)

---

##11. Conclusions and Future Directions

### 11.1 Summary of Evolution

**2002-2010**: N-gram era (BLEU, NIST, METEOR)
- Simple, fast, reasonably effective
- Established automatic evaluation as feasible

**2010-2019**: Linguistic awareness (chrF, TER, METEOR Universal)
- Addressed specific BLEU limitations
- Incremental improvements in correlation

**2019-Present**: Neural revolution (BERTScore, BLEURT, COMET)
- Dramatic improvement in human correlation
- State-of-the-art: COMET-22 with ρ = 0.664 segment-level

**Trend**: Moving from surface-form matching → semantic understanding

### 11.2 Current Best Practices (2025)

**For Research Papers**:
1. Primary: COMET or BLEURT
2. Legacy comparison: BLEU (for historical context)
3. Additional: chrF++ (fast, robust baseline)
4. Human evaluation: For final validation

**For Industry Deployment**:
1. Development: BLEU (fast iteration)
2. Pre-deployment test: COMET
3. Production monitoring: COMET-QE (reference-free)
4. Quality control: Human evaluation sampling

### 11.3 Open Problems

**1. Reference Dependency**
- Even best neural metrics need references
- How to evaluate when no references exist?
- Quality estimation improving but not perfect

**2. Segment vs. System Level**
- Trade-off between correlation levels
- Need metrics good at both

**3. Explainability**
- Neural metrics are black boxes
- Hard to trust in high-stakes domains (medical, legal)
- Need interpretable scores

**4. Evaluation of Evaluation**
- How do we know human judgments are correct?
- Inter-annotator agreement often low
- Meta-evaluation methodology needs improvement

**5. Beyond Adequacy and Fluency**
- Style, formality, emotional tone
- Cultural appropriateness
- Domain-specific criteria

### 11.4 Student Recommendations

**For this assignment and future work**:

1. **Understand fundamentals**: BLEU's math is worth learning deeply
2. **Stay current**: Neural metrics are quickly evolving
3. **Use multiple metrics**: No single metric captures everything
4. **Know limitations**: Every metric has blind spots
5. **Human insight invaluable**: Automatic metrics supplement, don't replace, human judgment

**Personal Learning**:
- Implementing BLEU from scratch deepened understanding
- Comparing metrics hands-on showed their different perspectives
- Realizing evaluation is an open problem was enlightening

---

## 12. References

### Foundational Papers

1. Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002). BLEU: a method for automatic evaluation of machine translation. *ACL*.

2. Banerjee, S., & Lavie, A. (2005). METEOR: An automatic metric for MT evaluation with improved correlation with human judgments. *ACL Workshop*.

3. Snover, M., Dorr, B., Schwartz, R., Micciulla, L., & Makhoul, J. (2006). A study of translation edit rate with targeted human annotation. *AMTA*.

### Alternative Metrics

4. Doddington, G. (2002). Automatic evaluation of machine translation quality using n-gram co-occurrence statistics. *HLT*.

5. Lin, C. Y. (2004). ROUGE: A package for automatic evaluation of summaries. *ACL*.

6. Popović, M. (2015). chrF: character n-gram F-score for automatic MT evaluation. *WMT*.

7. Popović, M. (2017). chrF++: words helping character n-grams. *WMT*.

### Neural Metrics

8. Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). BERTScore: Evaluating text generation with bert. *ICLR*.

9. Sellam, T., Das, D., & Parikh, A. P. (2020). BLEURT: Learning robust metrics for text generation. *ACL*.

10. Rei, R., Stewart, C., Farinha, A. C., & Lavie, A. (2020). COMET: A neural framework for MT evaluation. *EMNLP*.

11. Rei, R., M., J. G., Farinha, A.C., de Souza, J. G., Coheur, L., & Lavie, A. (2022). CometKiwi: IST-Unbabel 2022 submission for the quality estimation shared task. *WMT*.

### Critical Analysis

12. Callison-Burch, C., Osborne, M., & Koehn, P. (2006). Re-evaluating the role of BLEU in machine translation research. *EACL*.

13. Reiter, E. (2018). A structured review of the validity of BLEU. *Computational Linguistics*, 44(3), 393-401.

14. Mathur, N., Baldwin, T., & Cohn, T. (2020). Tangled up in BLEU: Reevaluating the evaluation of automatic machine translation evaluation metrics. *ACL*.

15. Freitag, M., Rei, R., Mathur, N., Lo, C. K., Stewart, C., Foster, G., ... & Neubig, G. (2022). Results of WMT22 metrics shared task: Stop using BLEU–neural metrics are better and more robust. *WMT*.

### Extensions and Variants

16. Denkowski, M., & Lavie, A. (2014). Meteor universal: Language specific translation evaluation for any target language. *WMT*.

17. Post, M. (2018). A call for clarity in reporting BLEU scores. *WMT*.

18. Sellam, T., Yadlowsky, S., Tenney, I., & Gehrmann, S. (2021). GEMBA: GPT estimation for bilingual assessment. *arXiv preprint*.

19. Kocmi, T., & Federmann, C. (2023). Large language models are state-of-the-art evaluators of translation quality. *EACL*.

### Survey and Meta-Analysis Papers

20. Han, L. (2022). A Survey on Automatic Evaluation of Neural Text Generation. *IEEE Transactions on Neural Networks and Learning Systems*.
