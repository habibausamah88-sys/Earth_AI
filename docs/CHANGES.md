# Revision notes / AI-assistance disclosure

This revision was prepared with AI coding assistance from the supplied project. The team should review every change, test it, and disclose assistance in the report.

- Official RGB dataset download filename corrected; nested archives supported; corrupted and exact decoded-pixel duplicate images excluded. Cross-label duplicate groups are discarded. Split seed fixed at 42, separate from training seed. Dataset signature recorded with each run. This is not proof of geographic independence.
- One-hot labels and categorical cross-entropy; same image decoding for training and prediction, including PNG.
- Model C loads a trained Model B on the matching dataset, unfreezes conv5 and keeps batch normalization frozen. Model A uses He initialization, dropout and optional batch normalization.
- All tuning arms actually change one named hyperparameter; four ablations, per-class recall, and three-seed validation summary; full selection by macro-F1 excludes smoke/capped models. Deployment uses the first predetermined final seed, not the best test seed.
- Learned baseline saved via shared trainer; final test stage evaluates both baselines and selected final model, optionally best validation model of each architecture. Generates separate metrics and failure artifacts. Test stage must not be used for tuning.
- Manual parameter formulas, model summaries, optional Graphviz architecture diagrams, GradientTape loss CSV, learning curves, manifest and experiment log.
- Original CSV retained as legacy_unverified_log.csv; not used in new selection. No fake results.
- TensorFlow is absent from the authoring environment; syntax checks and isolated data/metric checks are not training verification. The Colab notebook is the runtime verification path. No raw dataset or trained weights are bundled.
