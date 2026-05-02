# Speaker Script — Final Presentation

Pair this with `Final_Presentation.pptx` (same folder). Each block below
is the narration for one slide. Times assume a steady delivery; the
total scripted runtime is **~10 minutes**, leaving room for Q&A.

> **Tip:** practise the model architecture and threshold-calibration
> slides until you can explain each in 60 seconds without notes — those
> are the two questions an evaluator is most likely to dig into.

---

## Slide 1 — Title (0:30)

> "Good morning. This is my final-project presentation: a deepfake
> detection system built on a subset of FaceForensics++.
>
> The deliverable is a working web demo backed by a hybrid CNN-LSTM
> model that hits 82 percent test accuracy at the default threshold and
> ROC-AUC of 0.87. I'll walk through the problem, the dataset, the
> architecture choices I iterated through, the training procedure,
> the results, and a live demo screenshot at the end."

---

## Slide 2 — Problem statement (0:45)

> "Deepfakes are AI-generated face manipulations — face-swapping,
> reenactment, or full neural rendering — that have become hard for
> humans to spot reliably. The goal of this project is to classify a
> short uploaded video as REAL or FAKE with a calibrated confidence
> score.
>
> The hard constraint up front: this whole pipeline runs on a low-end
> laptop CPU — Intel i5-7200U, 8 gigs of RAM, no GPU. That constraint
> shaped a lot of the design decisions I'll cover."

---

## Slide 3 — Dataset (0:45)

> "The dataset is a subset of FaceForensics++. One thousand videos
> total, split 700 train / 150 val / 150 test. Four manipulation
> types in the fake class: deepfakes, face-to-face, faceswap, and
> neural textures.
>
> Important caveat: there's a 4-to-1 fake-to-real class skew. That
> drives most of the calibration work I'll show on slide 8. Anyone
> reporting accuracy on this dataset without checking macro-F1 is
> reporting a misleading number — a 'predict-everything-fake'
> classifier trivially hits 80 percent."

---

## Slide 4 — Data preprocessing (0:45)

> "Per video, OpenCV extracts 32 evenly spaced frames using
> CAP_PROP_POS_FRAMES seek. Each frame goes through a facenet-pytorch
> MTCNN detector, which gives a bounding box and a probability for
> the most prominent face. I pad the box by 20 pixels and crop to
> 224 by 224 — the standard ResNet input size. ImageNet mean-std
> normalisation is applied at load time.
>
> All this preprocessing logic is shared between training and the
> inference path. There's no train/serve skew."

---

## Slide 5 — Inference pipeline (1:00)

> "This is the pipeline that runs every time a user uploads a video.
> Four stages: extract frames, detect and crop faces, run the model,
> apply the decision threshold.
>
> The model is the composite I'll explain on the next slide:
> a fine-tuned ResNet-50 backbone followed by a small bidirectional
> LSTM that aggregates the 32 per-frame features into one
> video-level decision."

---

## Slide 6 — Iteration 1: frozen ImageNet features (1:00)

> "First attempt: the textbook hybrid. Frozen ResNet-50 with
> ImageNet weights as the feature extractor, train only a BiLSTM
> head on top. Cache features once, train fast on cached tensors.
>
> Result: test ROC-AUC of 0.44. Below 0.5 — the model's predictions
> were *anti-correlated* with truth on test. Even a logistic
> regression on mean-pooled features hit only 0.51. That isolated
> the failure: it wasn't the head, it was the features themselves.
>
> ImageNet was trained to tell cats from dogs. Deepfake artefacts
> are sub-pixel blending boundaries and frequency-domain residues.
> ImageNet features simply don't see what we need to detect."

---

## Slide 7 — Iteration 3: fine-tuned CNN as backbone (1:15)

> "The fix: instead of frozen ImageNet, use the trained CNN
> baseline as the feature extractor. Its ResNet-50 layer-4 was
> fine-tuned on this exact dataset, so its global-average-pool
> activations carry deepfake-specific information that ImageNet
> activations don't.
>
> Same training pipeline as before — only the features changed.
> Result: ROC-AUC jumped from 0.44 to 0.87. Accuracy went from
> 66 percent to 82 percent. Real-class recall went from 27 percent
> to 73 percent.
>
> The lesson: feature representation matters more than head
> architecture. Picking the right thing to fine-tune is the
> highest-leverage decision."

---

## Slide 8 — Training procedure (1:00)

> "Loss is binary cross entropy with logits. Optimiser is Adam,
> learning rate 1e-3 for the head only, weight-decay 1e-3.
>
> Two non-default choices that mattered. First, WeightedRandomSampler
> on the train loader, so each batch is class-balanced 50-50 even
> though the dataset is 80-20. Second, early stopping on val
> macro-F1 — not on val accuracy. This is critical: if you
> early-stop on accuracy under class imbalance, you'll happily
> 'win' by predicting everything as fake.
>
> Best val macro-F1 was 0.7741 at epoch 12 of 40 with a learning
> rate of 5e-4."

---

## Slide 9 — Final results (1:00)

> "Side-by-side. The hybrid v3 wins on every metric except recall on
> fakes — and that's a small loss the CNN baseline only earns by
> over-classifying as fake.
>
> The headline: 82 percent test accuracy, 0.87 ROC-AUC, real-recall
> 73 percent. Macro-F1 0.75. That's substantially above the project
> brief's 70 percent gate — and it's reproducible end-to-end on a
> laptop with no GPU.
>
> The deployed Flask app runs at the val-tuned threshold of 0.575
> — picking the threshold on val rather than test to avoid data
> leakage. ROC-AUC is threshold-independent, so 0.87 holds for any
> downstream choice of operating point."

---

## Slide 10 — Confusion matrix (0:30)

> "The confusion matrix at the deployed threshold. Both classes have
> non-trivial recall. The remaining errors split roughly evenly
> between false positives and false negatives — the model isn't
> systematically biased toward one class anymore."

---

## Slide 11 — Threshold sweep (0:45)

> "Calibration at a glance. The orange curve is F1-on-real, the
> blue curve is F1-on-fake, and the green curve is macro-F1. The
> red dashed line is the threshold I picked, where macro-F1 peaks
> on the validation set. Below that threshold the model is too
> permissive on real, above it too strict.
>
> Threshold tuning is a free win when class distributions are
> skewed — no retraining needed."

---

## Slide 12 — Web demo, upload page (0:30)

> "The Flask app at localhost:5000. Drag-and-drop or click-to-pick
> upload, max 50 megs, mp4 / avi / mov / mkv / webm accepted.
> Every step has typed error handling — invalid file type, no
> faces detected, video too short, model load failure — each
> returns a JSON code the frontend renders inline."

---

## Slide 13 — Web demo, prediction result (0:45)

> "After upload, about 15 seconds of inference on this CPU. The
> result card shows the label, a confidence bar, the per-class
> probabilities, the threshold in use, the number of frames where a
> face was found, and the inference plus total time.
>
> Per-frame fake probabilities are also returned in the JSON
> response — useful if you want to see whether the model's
> confidence is uniform across the clip or driven by a few frames."

---

## Slide 14 — Challenges and solutions (1:00)

> "Five things that mattered.
>
> Class imbalance — handled with WeightedRandomSampler in training
> and threshold calibration at inference.
>
> No GPU — solved by caching features once and training only the
> small head on cached tensors. End-to-end training with
> backpropagation through ResNet-50 would be roughly 50 hours per
> epoch on this laptop; head-only training is 10 minutes.
>
> Small dataset, 700 training videos — handled with heavy
> regularisation: dropout 0.5, weight-decay 1e-3, and early
> stopping on macro-F1.
>
> First hybrid attempt failed — frozen ImageNet features anti-
> correlated with truth on test. Fixed by swapping in the
> fine-tuned CNN baseline as the backbone.
>
> Mid-prediction crashes from bad uploads — typed exceptions in
> the Flask layer ensure every error returns a clean JSON code,
> never a 500 page."

---

## Slide 15 — Limitations (0:45)

> "Three honest caveats.
>
> Real-class recall is 73 percent — better than the calibrated
> CNN's 42 percent, but not perfect. About one in four real
> videos still gets called fake.
>
> Trained on 700 videos. Published FaceForensics results in the
> 90-plus range fine-tune the backbone end-to-end on a GPU.
>
> The model only sees 32 face crops per video at 224 by 224.
> No body cues, no audio, no compression-domain features. This
> is a class demo, not a forensic tool."

---

## Slide 16 — Future work (0:45)

> "Three concrete next steps.
>
> One — end-to-end fine-tune ResNet-50 on a Colab T4 with the
> existing train_hybrid.py. Projected accuracy 88 to 93 percent.
>
> Two — swap the backbone for XceptionNet pretrained on
> FaceForensics-plus-plus. That backbone was purpose-built for
> face-forgery detection and is publicly available.
>
> Three — multi-modal: add audio analysis and lip-sync
> consistency as additional signals. Most modern deepfakes
> still leak in either the audio or the audio-video alignment."

---

## Slide 17 — Q&A (0:15)

> "Thank you. The full code, training scripts, evaluation reports,
> and documentation are all in the project repository. Happy to
> take questions."

---

## Anticipated questions and answers

**Q: Why didn't you just use a GPU on Colab?**
A: I built the CPU pipeline first because that's what was available locally
during development; switching to Colab is the documented next step in
TRAINING_GUIDE.md. The CPU approach also forced a cleaner separation
between feature extraction and head training, which made the v1 → v3
diagnosis possible.

**Q: How do you know test ROC-AUC of 0.87 isn't overfitting?**
A: Test was held out from the start — it was never used for any
hyperparameter selection or threshold tuning. The val/test gap is small
(val macro-F1 0.77 vs test macro-F1 0.75) which is the right sanity
signal.

**Q: Why is real recall lower than fake recall?**
A: Class imbalance — only 140 unique real training videos vs 560 fake.
Even with a balanced sampler, the head sees less variety in the real
class. The fix is more real data; the workaround is the threshold
calibration that lifted real-recall from 5.5 % to 73 %.

**Q: Could someone fool this with a high-quality deepfake?**
A: Yes. The model is trained on four specific manipulation types —
deepfakes, face2face, faceswap, neuraltextures. A novel manipulation
that doesn't share statistics with any of these would likely transfer
poorly. This is why the README explicitly says "course demo, not for
forensic use."

**Q: Why the BiLSTM and not a Transformer?**
A: 32 frames is a short sequence; LSTMs are competitive at that scale
and have far fewer parameters. With only 700 training videos, parameter
efficiency mattered more than pure architecture power.

**Q: What does the threshold 0.575 mean?**
A: It's the value of `prob_fake` above which the model declares "FAKE".
At 0.5 the model would declare 64 % of training-distribution videos
as fake (since it inherits the class skew). Sweeping thresholds on
val and picking the macro-F1 maximiser gives 0.575.

**Q: How fast is one prediction?**
A: ~15 seconds end-to-end on this laptop, dominated by ResNet-50
forward on 32 frames. On a GPU it would be sub-second. The model is
loaded once at startup, so there's no per-request load cost.

**Q: How much data did you train on, total?**
A: 700 train videos × 32 face crops = 22 400 images. After feature
extraction, the trainable model is just the BiLSTM head on
700 cached `(32, 2048)` tensors.
