# Which Distortion Gets Confused? Cognitive Distortion Classification and Error Analysis

Code, results and predictions for the paper **"Which Distortion gets confused? A Controlled Comparison of Language Models for Dominant Cognitive Distortion Classification"** (Rishi Gnanasekar and Caleb Siefert, University of Michigan-Dearborn).

We compare TF-IDF baselines with RoBERTa, DeBERTa-v3 and MentalBERT (with and without weighted focal loss) on dominant cognitive distortion classification, using the annotated distorted span as input. We report results over three training seeds, significance tests, per-class scores and an analysis of which distortions are confused.

## Task and data

- **Dataset:** Therapist Q&A dataset (Shreevastava and Foltz, 2021). We drop "No Distortion" messages and keep 1,597 distorted messages with 10 dominant-distortion classes.
- **Input:** the annotated distorted span.
- **Split:** stratified test set of 240 messages (split seed 3407) and 1,357 messages for training and validation. The same split is used for every model.
- **Training seeds:** 1, 2 and 3 (separate from the split seed).
- The dataset is not included in this repository. Get it from the original authors and set `DATA_PATH` at the top of the notebook (the notebook expects `Annotated_data.csv`).

## Repository contents

| File or folder | What it does |
|---|---|
| `Cognitive_Distortion_All_Models_3Seeds (2).ipynb` | Main notebook. Trains and evaluates all six models over three seeds, then computes the results table, significance tests, per-class F1, confusion matrices and top confusions. |
| `Roberta_Cross_entropy (1).ipynb` | 5-fold cross-validation and final run for RoBERTa-base. |
| `Deberta_Cross_entropy (1).ipynb` | 5-fold cross-validation and final run for DeBERTa-v3-base. |
| `Mental_Bert_Cross_entropy (1).ipynb` | 5-fold cross-validation and final run for MentalBERT (cross-entropy). |
| `results/` | Outputs of the main notebook (see below). |

## Main results (test set, mean over three seeds)

| Model | Accuracy | Macro F1 |
|---|---|---|
| MentalBERT (CE) | 0.442 | 0.406 |
| RoBERTa-base | 0.439 | 0.393 |
| DeBERTa-v3-base | 0.415 | 0.376 |
| MentalBERT (focal) | 0.392 | 0.355 |
| TF-IDF + LR | 0.367 | 0.331 |
| TF-IDF + RF | 0.292 | 0.236 |

The majority-class baseline reaches 0.15 accuracy. The three cross-entropy encoders are not significantly different from each other, and weighted focal loss is worse. See the paper for standard deviations, kappa, McNemar tests and the paired bootstrap.

## How to run

1. Open `Cognitive_Distortion_All_Models_3Seeds (2).ipynb` in Google Colab with a GPU runtime.
2. Set `DATA_PATH` in the first code cell to your copy of the dataset.
3. Run the cells in order. Training is resumable: predictions are saved to `results/preds/` as they finish, and finished runs are skipped on a rerun.
4. The Colab folder `results/` is temporary storage. Download it (or mount Google Drive) before the runtime resets.

The cross-validation notebooks use 6 epochs per fold. Final encoder runs train for up to 10 epochs (15 for DeBERTa-v3) with 10% of the training data held out for early stopping on macro F1 (patience 3).

## Output files in `results/`

| File | Content |
|---|---|
| `preds/` (inside `results/`) | Saved test predictions for every model and seed (`.npy`). |
| `y_test.npy`, `classes.json` | Gold test labels and class names. |
| `results_table.csv` | Accuracy, macro F1, weighted F1 and kappa per model (mean and standard deviation). |
| `significance.csv` | Pairwise paired bootstrap intervals and McNemar p-values per seed. |
| `per_class_f1.csv` | Per-class F1 for each model. |
| `confusion_*.png` | Row-normalised confusion matrix per encoder. |
| `confusion_mean_encoders.png` | Confusion matrix averaged over the four encoders (Figure 1 in the paper). |
| `confusion_similarity.csv` | Similarity of the confusion patterns between encoders. |
| `top_confusions.csv`, `top_confusions_mean.csv` | Most frequent confusions for one model and for the encoder average (Table 5 in the paper). |

## Notes

- Fine-tuning on about 1,350 messages is noisy and GPU runs are not fully deterministic, so a full rerun can shift results by a few points.
- The weighted focal variant also uses class weights, so any difference from cross-entropy mixes both effects.
- Do not commit access tokens. Set your Hugging Face token as a Colab secret or environment variable, not in a notebook cell.

## Citation

If you use this work, please cite the paper (citation to be added after publication).

## Acknowledgement of data source

Shreevastava, S. and Foltz, P. (2021). Detecting cognitive distortions from patient-therapist interactions. *Proceedings of the Seventh Workshop on Computational Linguistics and Clinical Psychology*, pages 151-158.
