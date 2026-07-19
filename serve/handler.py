import json

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler


class EmbeddingHandler(BaseHandler):
    """bge-m3-address SentenceTransformer(mean pooling + L2 normalize) equivalent handler."""

    def initialize(self, context):
        model_dir = context.system_properties.get("model_dir")
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True

    def preprocess(self, requests):
        texts = []
        for req in requests:
            body = req.get("body") or req.get("data")
            if isinstance(body, (bytes, bytearray)):
                body = body.decode("utf-8")
            if isinstance(body, str):
                body = json.loads(body)
            if isinstance(body, dict):
                texts.append(body.get("text") or body.get("texts"))
            else:
                texts.append(body)
        return texts

    def inference(self, texts):
        flat_texts = []
        spans = []
        for t in texts:
            if isinstance(t, list):
                flat_texts.extend(t)
                spans.append(len(t))
            else:
                flat_texts.append(t)
                spans.append(1)

        encoded = self.tokenizer(
            flat_texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**encoded)
            token_embeddings = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (token_embeddings * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            embeddings = summed / counts
            embeddings = F.normalize(embeddings, p=2, dim=1)

        results = []
        idx = 0
        for span in spans:
            chunk = embeddings[idx: idx + span].tolist()
            results.append(chunk if span > 1 else chunk[0])
            idx += span
        return results

    def postprocess(self, inference_output):
        return inference_output
