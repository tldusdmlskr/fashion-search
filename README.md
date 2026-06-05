# Fashion Search - Retrieval Pipeline

## 개요
자연어 쿼리 기반 패션 이미지 검색 시스템의 리트리벌 파이프라인입니다.

## 파이프라인 구조
자연어 쿼리
↓
fashion_dict 치환 → Google 번역
↓
marqo-fashionSigLIP으로 쿼리 임베딩
↓
Qdrant Visual Index Top-100 검색
↓
카테고리 사전 필터링
↓
캡션 키워드 부스트/페널티 재정렬
↓
ko-sroberta 의미적 리랭킹
↓
Top-10 반환
