from pathlib import Path
from infer import SkinCancerDetector

model = SkinCancerDetector('./model_artifacts')
files = sorted(Path('../sorted_by_dx/bcc').glob('*.jpg'))[:50]
correct = 0
for p in files:
    r = model.predict(str(p))
    if r.get('accepted') and r.get('predicted_class') == 'bcc':
        correct += 1

print('samples:', len(files))
print('predicted_bcc:', correct)
print('sample_accuracy:', round(correct / len(files), 4) if files else 0)
