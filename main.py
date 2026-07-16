import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


MODEL_NAME = "BAAI/bge-m3"


def mean_pooling(model_output, attention_mask):
    """
    Transformer 출력값(token embedding)을
    하나의 문장 embedding으로 변환

    Args:
        model_output:
            last_hidden_state
            shape: [batch, token, hidden]

        attention_mask:
            실제 토큰 여부
            shape: [batch, token]

    Returns:
        sentence_embedding
            shape: [batch, hidden]
    """

    # Transformer 최종 출력
    token_embeddings = model_output.last_hidden_state

    # [batch, token]
    # ->
    # [batch, token, 1]
    #
    # token embedding과 곱하기 위한 형태로 변경
    mask = attention_mask.unsqueeze(-1)

    # Padding token 제거
    # 실제 token: embedding * 1
    # padding:    embedding * 0
    masked_embeddings = token_embeddings * mask

    # token 방향(dim=1)으로 합산
    # [batch, token, hidden]
    # ->
    # [batch, hidden]
    sum_embeddings = masked_embeddings.sum(dim=1)

    # 실제 token 개수 계산
    # 평균을 내기 위한 분모
    sum_mask = mask.sum(dim=1)

    # Mean Pooling
    sentence_embedding = sum_embeddings / sum_mask

    return sentence_embedding


def get_embedding(texts, tokenizer, model):
    """
    문장을 입력받아 BGE-M3 문장 벡터 생성

    Args:
        texts:
            문자열 또는 문자열 리스트

    Returns:
        sentence embedding
        shape: [batch, 1024]
    """

    # 문자열 하나가 들어와도 batch 형태로 처리
    if isinstance(texts, str):
        texts = [texts]

    # Tokenizer
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # 추론 모드
    # gradient 계산하지 않음
    with torch.no_grad():
        outputs = model(**inputs)

    # Token embedding -> Sentence embedding
    embeddings = mean_pooling(
        outputs,
        inputs["attention_mask"]
    )

    # Cosine similarity 계산을 위해 정규화
    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=1
    )

    return embeddings


# ==========================
# Model Load
# ==========================

print("Tokenizer Loading...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Model Loading...")
model = AutoModel.from_pretrained(MODEL_NAME)

model.eval()


# ==========================
# Embedding 생성 테스트
# ==========================

texts = [
    "서울특별시 강남구 테헤란로 20",
    "서울특별시 구로구 가마산로 20"
]


embeddings = get_embedding(
    texts,
    tokenizer,
    model
)


print("Embedding Shape")
print(embeddings.shape)
# [2,1024]
#
# 의미:
# 문장 2개
# 각 문장을 1024차원 벡터로 변환


embedding1 = embeddings[0]
embedding2 = embeddings[1]


# ==========================
# Cosine Similarity
# ==========================

similarity = F.cosine_similarity(
    embedding1,
    embedding2,
    dim=0
)


print("Cosine Similarity:")
print(similarity.item())