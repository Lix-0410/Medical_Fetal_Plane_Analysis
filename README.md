# Fetal Plane Classifier

A two-stage deep learning pipeline that classifies fetal ultrasound images by anatomical plane, built with an emphasis on interpretability rather than raw accuracy alone. Given a single ultrasound frame, the system first determines whether it shows a clean, standard anatomical view at all — and only if so, identifies which of seven anatomical planes it represents. Every prediction is paired with a Grad-CAM heatmap so the model's reasoning can be visually inspected rather than taken on faith.

This project was built end-to-end: dataset preprocessing, model design, training, evaluation, interpretability analysis, and deployment as a Flask web app.

---

## Why two stages instead of one

The dataset (FETAL_PLANES_DB) labels each image as one of six anatomical categories or "Other" — a catch-all for transitional, partial, or non-diagnostic frames captured while the sonographer moves the probe between standard views.

A single model predicting all eight labels at once has to learn two fundamentally different judgments simultaneously: *is there a usable anatomical landmark here at all*, and *if so, which one*. These pull feature learning in different directions and make failures hard to diagnose — a wrong prediction could mean either "missed a real plane" or "confused two real planes," with no way to tell which from the output alone.

Splitting this into two models mirrors how a sonographer actually works, and makes errors interpretable:

```
                    Input ultrasound frame
                              │
                    ┌─────────▼──────────┐
                    │   STAGE 1            │
                    │   ResNet50 (binary)  │
                    │   Standard vs Other   │
                    └─────────┬───────────┘
                              │
              ┌───────────────┴────────────────┐
              │                                  │
          "Other"                          "Standard"
              │                                  │
              ▼                                  ▼
      Final: Other                    ┌──────────────────────┐
      (no further                     │   STAGE 2              │
       processing)                    │   ResNet50 (7-class)   │
                                       │   Anatomy classifier    │
                                       └──────────┬─────────────┘
                                                  │
                                       One of 7 standard planes:
                                       Brain (×3 sub-planes),
                                       Abdomen, Femur, Thorax, Cervix
```

---

## Dataset

**FETAL_PLANES_DB** — 12,400 ultrasound images from 1,792 patients, collected across three ultrasound machines (Voluson E6, Voluson S10, Aloka) by multiple operators.

| Class | Images | Notes |
|---|---|---|
| Other | 4,356 | Non-standard / transitional frames (includes 143 ambiguous brain sweeps, merged in) |
| Fetal thorax | 1,718 | |
| Brain — Trans-thalamic | 1,638 | Primary biometric brain plane |
| Maternal cervix | 1,626 | |
| Fetal femur | 1,040 | |
| Brain — Trans-cerebellum | 714 | |
| Fetal abdomen | 711 | Smallest standard class |
| Brain — Trans-ventricular | 597 | |

### Splitting methodology

- The dataset's original train/test split was preserved as-is.
- A validation set was carved out of the training pool at **patient level** (not image level) to prevent leakage — multiple frames from the same patient never appear across different splits.
- An 85/15 split was used instead of the more typical 80/20, because the smallest class (Fetal abdomen) had only 9 patients in the training pool; 85/15 was the minimum adjustment needed to guarantee every class had representation in validation.
- Zero patient overlap between train, validation, and test was explicitly verified.

---

## Models

Both stages use **ResNet50**, pretrained on ImageNet and fine-tuned end-to-end (no frozen layers). ResNet50 was chosen specifically because Grad-CAM — the interpretability method used throughout this project — was originally developed and validated on ResNet-style architectures, giving the most reliable heatmaps of the architectures considered.

**Stage 1 — Standard vs. Other (binary)**
- `BCEWithLogitsLoss` with `pos_weight` to correct for class imbalance
- Trained with early stopping (patience = 4 epochs) on validation loss
- Test accuracy: **92.16%**

**Stage 2 — Anatomy classification (7-class)**
- `CrossEntropyLoss` with per-class weights, macro-averaged metrics throughout (so the smallest class, Trans-ventricular, isn't drowned out by larger ones in reported performance)
- Same training regimen as Stage 1
- Test accuracy: **92.65%**

---

## Results

### Stage 1 — Standard vs Other

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Other | 0.884 | 0.868 | 0.876 |
| Standard | 0.939 | 0.947 | 0.943 |

The decision threshold (0.5) was validated by sweeping from 0.30 to 0.70 and confirming it maximizes both overall accuracy and macro F1 — the imbalance in Other-class recall is not a tuning artifact, it reflects a genuine, harder decision boundary for this class.

### Stage 2 — Anatomy classification

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Fetal femur | 1.000 | 0.994 | 0.997 |
| Maternal cervix | 1.000 | 0.995 | 0.998 |
| Fetal thorax | 0.991 | 0.983 | 0.987 |
| Fetal abdomen | 0.978 | 0.989 | 0.983 |
| Brain — Trans-thalamic | 0.810 | 0.893 | 0.850 |
| Brain — Trans-cerebellum | 0.859 | 0.788 | 0.822 |
| Brain — Trans-ventricular | 0.822 | 0.705 | 0.759 |

Four of seven classes are at or near ceiling performance. The three brain sub-planes are where the model struggles — particularly **Trans-ventricular**, confused with Trans-thalamic in roughly 29% of cases.

---

## Interpretability: what Grad-CAM showed

Every prediction in the deployed app is accompanied by a Grad-CAM heatmap, generated by hooking into ResNet50's final convolutional block.

**Finding 1 — Confusion is anatomically grounded, not random.** When the model misclassifies a Trans-ventricular frame as Trans-thalamic, the heatmap still activates over the correct general brain region — it isn't looking at noise, text overlays, or irrelevant structures. The error comes from fine-grained discrimination within the right neighborhood, not from looking in the wrong place.

**Finding 2 — Initial concern, resolved.** Early on, nearly every correctly classified standard image showed a similarly shaped, centered activation blob, raising a concern that the model might be relying on "stuff is centered in frame" as a shortcut rather than genuine anatomical features. This was tested directly using Stage 1's behavior on true **Other** images: if the model were using a centering shortcut, it would still light up the center even when no real anatomy is present. Instead, Grad-CAM on Other images showed attention scattering to frame edges, corners, and on-screen artifacts — wherever the salient signal in that particular (non-diagnostic) frame happened to be. This confirmed the centering pattern in Stage 2 reflects real sonographer practice (anatomy of interest is centered during acquisition), not a model shortcut.

---

## Known limitations

- **Not validated for clinical use.** This is a research/portfolio project trained on a single public dataset.
- **Stage 1 misses ~13% of true non-standard frames**, passing them to Stage 2, which will then confidently assign them an anatomy label they don't actually show. The app surfaces Stage 1's confidence score so this can be judged by the user, but does not hide the underlying risk.
- **Brain sub-plane classification is the weakest part of the pipeline**, for the anatomical reasons described above.
---

## Project structure

```
fetal-plane-classifier/
├── app.py                 # Flask routes
├── inference.py           # Model loading, two-stage prediction, Grad-CAM
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   └── js/main.js
├── models/
│   ├── stage1_best_model.pth
│   └── stage2_best_model.pth
├── requirements.txt
└── README.md
```

## Running locally

```bash
git clone https://github.com/<your-username>/Medical_Fetal_Plane_Analysis.git
cd Medical_Fetal_Plane_Analysis
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:5000` and upload an ultrasound frame (PNG/JPG).

---

## Tech stack

PyTorch · torchvision (ResNet50) · Flask · Grad-CAM · scikit-learn (evaluation) · trained on Kaggle (T4/P100 GPU)

## Dataset citation

FETAL_PLANES_DB — publicly available fetal ultrasound plane classification dataset.

---

## License

MIT
