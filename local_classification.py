"""@misc{mozilla_iab_multilabel_lora,
  title       = {Fine-tuned LoRA Classifier on MiniLM for IAB Multi-Label Classification},
  author      = {Mozilla},
  year        = {2025},
  url         = {https://huggingface.co/mozilla/content-multilabel-iab-classifier},
  license     = {Apache-2.0}
}"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

tokenizer = AutoTokenizer.from_pretrained("Mozilla/content-multilabel-iab-classifier")
model = AutoModelForSequenceClassification.from_pretrained("Mozilla/content-multilabel-iab-classifier")

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

with torch.no_grad():
    for text in prompts_neg[0:10]:
        print(textcme)
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
        predicted_labels = [(id2label[i], round(p, 3)) for i, p in enumerate(probs) if p >= 0.5]

