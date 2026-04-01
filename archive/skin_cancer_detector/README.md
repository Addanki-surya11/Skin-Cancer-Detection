# Skin Cancer Detector (CNN + XGBoost)

This project trains a skin lesion classifier on HAM10000-style class folders using:

- CNN: ResNet18 (fine-tuned)
- XGBoost: final classifier on CNN embeddings
- Rejection gates to reject non-skin / out-of-distribution / low-confidence images

## Dataset format

Expected folder format:

- `sorted_by_dx/akiec/*.jpg`
- `sorted_by_dx/bcc/*.jpg`
- `sorted_by_dx/bkl/*.jpg`
- `sorted_by_dx/df/*.jpg`
- `sorted_by_dx/mel/*.jpg`
- `sorted_by_dx/nv/*.jpg`
- `sorted_by_dx/vasc/*.jpg`

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py --data-dir ../sorted_by_dx --output-dir ./model_artifacts --epochs 12 --batch-size 32
```

Artifacts saved in `model_artifacts/`:

- `cnn_model.pt`
- `xgb_model.json`
- `ood_isolation_forest.joblib`
- `metadata.json`

## Inference (CLI)

```bash
python infer.py --model-dir ./model_artifacts --image ../sorted_by_dx/nv/ISIC_0024306.jpg
```

## Streamlit app

```bash
streamlit run app.py
```

## Notes

- The script reports measured test accuracy; exact accuracy depends on available images and compute.
- If test accuracy is below 90%, improve by increasing epochs, using more source images, and trying a stronger CNN backbone.
