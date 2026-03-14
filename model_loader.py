import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

MODEL_DIR = "edos_export"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EDOSModel(nn.Module):
    def __init__(self, encoder_path: str, dropout_p: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path)
        h = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout_p)
        self.head_a = nn.Linear(h, 2)
        self.head_b = nn.Linear(h, 5)
        self.head_c = nn.Linear(h, 11)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.drop(out.last_hidden_state[:, 0, :])
        return self.head_a(cls), self.head_b(cls), self.head_c(cls)


class ModelService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.inv_task_a = None
        self.inv_task_b = None
        self.inv_task_c = None

    def load(self):
        config_path = os.path.join(MODEL_DIR, "config.json")
        with open(config_path, "r") as f:
            self.config = json.load(f)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_DIR,
                fix_mistral_regex=True
            )
        except TypeError:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

        encoder_path = os.path.join(MODEL_DIR, self.config["encoder_dir"])
        model = EDOSModel(
            encoder_path=encoder_path,
            dropout_p=self.config["dropout_p"]
        )

        heads_path = os.path.join(MODEL_DIR, "custom_heads.pt")
        heads_ckpt = torch.load(heads_path, map_location=DEVICE)

        model.head_a.load_state_dict(heads_ckpt["head_a_state_dict"])
        model.head_b.load_state_dict(heads_ckpt["head_b_state_dict"])
        model.head_c.load_state_dict(heads_ckpt["head_c_state_dict"])

        model.to(DEVICE)
        model.eval()

        self.model = model
        self.inv_task_a = {v: k for k, v in self.config["task_a_labels"].items()}
        self.inv_task_b = {v: k for k, v in self.config["task_b_labels"].items()}
        self.inv_task_c = {v: k for k, v in self.config["task_c_labels"].items()}

    def predict(self, text: str):
        if self.model is None or self.tokenizer is None or self.config is None:
            raise RuntimeError("Model service not loaded")

        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.config["max_len"],
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)

        with torch.no_grad():
            logits_a, logits_b, logits_c = self.model(input_ids, attention_mask)

            pred_a = int(torch.argmax(logits_a, dim=1).item())
            pred_b = int(torch.argmax(logits_b, dim=1).item())
            pred_c = int(torch.argmax(logits_c, dim=1).item())

            prob_a = torch.softmax(logits_a, dim=1).cpu().numpy()[0].tolist()
            prob_b = torch.softmax(logits_b, dim=1).cpu().numpy()[0].tolist()
            prob_c = torch.softmax(logits_c, dim=1).cpu().numpy()[0].tolist()

        return {
            "task_a": {
                "id": pred_a,
                "label": self.inv_task_a[pred_a],
                "probs": prob_a
            },
            "task_b": {
                "id": pred_b,
                "label": self.inv_task_b[pred_b],
                "probs": prob_b
            },
            "task_c": {
                "id": pred_c,
                "label": self.inv_task_c[pred_c],
                "short_label": self.config["task_c_short"][pred_c],
                "probs": prob_c
            }
        }


model_service = ModelService()