"""@misc{mozilla_iab_multilabel_lora,
  title       = {Fine-tuned LoRA Classifier on MiniLM for IAB Multi-Label Classification},
  author      = {Mozilla},
  year        = {2025},
  url         = {https://huggingface.co/mozilla/content-multilabel-iab-classifier},
  license     = {Apache-2.0}
}"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from textwrap import fill

tokenizer = AutoTokenizer.from_pretrained("mozilla/content-multilabel-iab-classifier")
model = AutoModelForSequenceClassification.from_pretrained("mozilla/content-multilabel-iab-classifier")

label_list = [
    'inconclusive',
    'animals',
    'arts',
    'autos',
    'business',
    'career',
    'education',
    'fashion',
    'finance',
    'food',
    'government',
    'health',
    'hobbies',
    'home',
    'news',
    'realestate',
    'society',
    'sports',
    'tech',
    'travel'
]

label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

with open("prompts.json") as json_file :
    data = json.load(json_file)

prompts_neg = data["negative"]["prompts"]

probas = []

with torch.no_grad():
    for text in prompts_neg:
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
        predicted_labels = [(id2label[i], round(p, 3)) for i, p in enumerate(probs) if p >= 0.5]
        probas.append(probs)

probas = np.array(probas)
label_cpt = (probas>=0.5).sum(axis=0)

plt.figure()
plt.barh(label_list, label_cpt)
plt.xlabel("Quantite")
plt.ylabel("Labels")
plt.title("Distribution des labels predits")
plt.tight_layout()
plt.savefig("hist.png", dpi=300)

