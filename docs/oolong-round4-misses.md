# OOLONG round 4 (docdb + voting) — miss review

50 questions, 32 correct (64%). All 18 misses below with full question,
prediction, and gold for human verification. Trajectories (full code +
labeling calls) are in `runs/oolong4/oolong-docdb/trajectories/<qid>.jsonl`.

## 1. `oolong-12000044`  (sub-calls: 10, iterations: 3)

**Question:** In the above data, how many data points should be classified as label 'entity'? Give your final answer in the form 'Answer: number'.

**Predicted:** 11

**Gold:** 10

**Diagnosis:** Voted labels; count off by 1 (11 vs 10).

## 2. `oolong-14000016`  (sub-calls: 32, iterations: 4)

**Question:** In the above data, how many data points should be classified as label 'abbreviation'? Give your final answer in the form 'Answer: number'.

**Predicted:** 59

**Gold:** 64

**Diagnosis:** Voted labels; undercount (59 vs 64).

## 3. `oolong-15000239`  (sub-calls: 62, iterations: 3)

**Question:** In the above data, how many data points should be classified as label 'human being'? Give your final answer in the form 'Answer: number'.

**Predicted:** 145

**Gold:** 155

**Diagnosis:** Voted labels; undercount on a 655-line context (145 vs 155).

## 4. `oolong-15000248`  (sub-calls: 1, iterations: 2)

**Question:** For the following question, only consider the subset of instances that are associated with user IDs 62297. Among instances associated with these users, is label 'numeric value' more common, less common, or the same frequency as label 'abbreviation'? Give your final answer in the form 'Answer: numeric value is [X] abbreviation', where [X] is 'more common than', 'less common than', or 'same frequency as'.

**Predicted:** numeric value is same frequency as abbreviation

**Gold:** less common than

**Diagnosis:** Inherited stale labels from a sibling question (1 sub-call, no fresh annotation).

## 5. `oolong-10000025`  (sub-calls: 18, iterations: 6)

**Question:** In the above data, which of the labels is the least common? Give your final answer in the form 'Label: answer' where answer is one of the labels: entity, human being, description and abstract concept, location, numeric value, abbreviation.

**Predicted:** Label: location

**Gold:** description and abstract concept, location, numeric value, abbreviation

**Diagnosis:** TIE-SET FORMAT: predicted label IS in the gold tie set, but gold expects all tied labels listed.

## 6. `oolong-12000043`  (sub-calls: 1, iterations: 3)

**Question:** In the above data, is label 'location' more common, less common, or the same frequency as label 'description and abstract concept'? Give your final answer in the form 'Answer: location is [X] description and abstract concept', where [X] is 'more common than', 'less common than', or 'same frequency as'.

**Predicted:** location is less common than description and abstract concept

**Gold:** more common than

**Diagnosis:** Inherited stale labels; comparison direction flipped.

## 7. `oolong-10000041`  (sub-calls: 1, iterations: 3)

**Question:** In the above data, how many data points should be classified as label 'entity'? Give your final answer in the form 'Answer: number'.

**Predicted:** 5

**Gold:** 3

**Diagnosis:** Inherited stale labels; overcount (5 vs 3).

## 8. `oolong-11000048`  (sub-calls: 1, iterations: 4)

**Question:** In the above data, how many data points should be classified as label 'human being'? Give your final answer in the form 'Answer: number'.

**Predicted:** 6

**Gold:** 7

**Diagnosis:** Inherited stale labels; off by 1.

## 9. `oolong-13000051`  (sub-calls: 16, iterations: 4)

**Question:** In the above data, how many data points should be classified as label 'human being'? Give your final answer in the form 'Answer: number'.

**Predicted:** 28

**Gold:** 1

**Diagnosis:** OUTLIER (28 vs 1): likely misread question semantics — worth human review first.

## 10. `oolong-16000208`  (sub-calls: 124, iterations: 3)

**Question:** In the above data, is label 'description and abstract concept' more common, less common, or the same frequency as label 'human being'? Give your final answer in the form 'Answer: description and abstract concept is [X] human being', where [X] is 'more common than', 'less common than', or 'same frequency as'.

**Predicted:** Answer: description and abstract concept is more common than human being

**Gold:** less common than

**Diagnosis:** Voted labels; close-count comparison flipped.

## 11. `oolong-14000021`  (sub-calls: 1, iterations: 4)

**Question:** In the above data, how many data points should be classified as label 'numeric value'? Give your final answer in the form 'Answer: number'.

**Predicted:** 69

**Gold:** 68

**Diagnosis:** Inherited labels; off by 1 (69 vs 68).

## 12. `oolong-15000214`  (sub-calls: 1, iterations: 4)

**Question:** In the above data, how many data points should be classified as label 'location'? Give your final answer in the form 'Answer: number'.

**Predicted:** 114

**Gold:** 127

**Diagnosis:** Reused annotation (1 sub-call); undercount (114 vs 127).

## 13. `oolong-16000248`  (sub-calls: 1, iterations: 3)

**Question:** For the following question, only consider the subset of instances that are associated with user IDs 19439. Among instances associated with these users, which of the labels is the least common? Give your final answer in the form 'Label: answer' where answer is one of the labels: abbreviation, entity, description and abstract concept, numeric value, location, human being.

**Predicted:** Label: human being

**Gold:** abbreviation, numeric value, human being

**Diagnosis:** TIE-SET FORMAT: predicted label is in the gold set; full tie not listed.

## 14. `oolong-10000034`  (sub-calls: 1, iterations: 2)

**Question:** In the above data, is label 'human being' more common, less common, or the same frequency as label 'abbreviation'? Give your final answer in the form 'Answer: human being is [X] abbreviation', where [X] is 'more common than', 'less common than', or 'same frequency as'.

**Predicted:** Answer: human being is same frequency as abbreviation

**Gold:** more common than

**Diagnosis:** Inherited stale labels; tie called where gold says 'more common'.

## 15. `oolong-11000006`  (sub-calls: 8, iterations: 4)

**Question:** In the above data, is label 'description and abstract concept' more common, less common, or the same frequency as label 'numeric value'? Give your final answer in the form 'Answer: description and abstract concept is [X] numeric value', where [X] is 'more common than', 'less common than', or 'same frequency as'.

**Predicted:** Answer: description and abstract concept is more common than numeric value

**Gold:** same frequency as

**Diagnosis:** Voted labels; exact tie missed (called 'more common').

## 16. `oolong-13000040`  (sub-calls: 16, iterations: 3)

**Question:** In the above data, is label 'numeric value' more common, less common, or the same frequency as label 'abbreviation'? Give your final answer in the form 'Answer: numeric value is [X] abbreviation', where [X] is 'more common than', 'less common than', or 'same frequency as'.

**Predicted:** Answer: numeric value is more common than abbreviation

**Gold:** less common than

**Diagnosis:** Voted labels; close-count comparison flipped.

## 17. `oolong-16000211`  (sub-calls: 1, iterations: 5)

**Question:** In the above data, is label 'entity' more common, less common, or the same frequency as label 'abbreviation'? Give your final answer in the form 'Answer: entity is [X] abbreviation', where [X] is 'more common than', 'less common than', or 'same frequency as'.

**Predicted:** Answer: entity is more common than abbreviation

**Gold:** less common than

**Diagnosis:** Inherited stale labels; comparison flipped.

## 18. `oolong-10000006`  (sub-calls: 1, iterations: 4)

**Question:** In the above data, is label 'abbreviation' more common, less common, or the same frequency as label 'description and abstract concept'? Give your final answer in the form 'Answer: abbreviation is [X] description and abstract concept', where [X] is 'more common than', 'less common than', or 'same frequency as'.

**Predicted:** Answer: abbreviation is less common than description and abstract concept

**Gold:** more common than

**Diagnosis:** Inherited stale labels; comparison flipped.
