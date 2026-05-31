

# 0. 드라이브 마운트 + Qdrant 복사

from google.colab import drive
drive.mount("/content/drive")

import os, shutil
if not os.path.exists("/content/qdrant_storage"):
    shutil.copytree(
        "/content/drive/.shortcut-targets-by-id/1bXw5JFGghpRs_YjLI_HMSuH7jpIusvoz/qdrant_storage",
        "/content/qdrant_storage"
    )
    print("복사 완료!")
else:
    print("이미 복사되어 있음!")


# 1. 모델 로드

import open_clip
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from deep_translator import GoogleTranslator
import torch
import time
import numpy as np
from IPython.display import Image, display

model, _, _ = open_clip.create_model_and_transforms("hf-hub:Marqo/marqo-fashionSigLIP")
clip_tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
qdrant = QdrantClient(path="/content/qdrant_storage")


# 2. 함수 정의


# 한국어 → 영어 번역
def translate_to_english(query):
    translated = GoogleTranslator(source="ko", target="en").translate(query)
    print(f"번역: {query} → {translated}")
    return translated

# 텍스트 → 임베딩 (정규화 포함)
def encode_text(query):
    query_en = translate_to_english(query)
    text_input = clip_tokenizer([query_en])
    with torch.no_grad():
        emb = model.encode_text(text_input)
    emb = emb.squeeze()
    emb = emb / emb.norm()  # 정규화
    return emb.numpy()

# Stage 1
def stage1_retrieve(query, top_k=10):
    query_image_emb = encode_text(query)
    visual_hits = qdrant.query_points(
        collection_name="visual",
        query=query_image_emb.tolist(),
        limit=top_k
    ).points
    return [(hit.id, hit.score) for hit in visual_hits]

# 이미지 포함 검색
def search_with_image(query, top_k=5):
    start = time.time()
    results = stage1_retrieve(query, top_k=top_k)
    print(f"검색 시간: {(time.time()-start)*1000:.0f}ms\n")
    for id_, score in results:
        payload = qdrant.retrieve(
            collection_name="visual",
            ids=[id_],
            with_payload=True
        )[0].payload
        filename = payload["filename"]
        url = f"https://pub-5966bf5d84f948c983500b6d9547eec9.r2.dev/masking_data/{filename}"
        print(f"ID: {id_} | 파일명: {filename} | 유사도: {score:.4f}")
        display(Image(url=url, width=200))


# 3. 테스트

search_with_image("루즈한 니트")
