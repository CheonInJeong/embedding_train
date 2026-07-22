# TODO: 번지수 근접 오답(hard negative) 학습 반영

## 배경

`docs/eval_report_db.md` 평가에서 파인튜닝 모델(`bge-m3-address`)이 틀리는 케이스는 대부분
"도로명은 맞는데 번지수만 1~2자리 다른 후보"였다 (예: 정답 `도곡로 404` → 예측 `도곡로 405`).
`MultipleNegativesRankingLoss`는 같은 배치 안의 다른 positive만 자동으로 negative로 쓰기 때문에,
이런 근접 오답을 구분하는 신호를 직접 받지 못한다.

## 지금까지 만든 것

- `scripts/build_hard_negatives.py`
  - `data/address/임베딩데이터(읍면리포함).txt`(92,702건)를 도로명 prefix로 인덱싱
  - 정답 주소와 번지수 차이가 1~5(`MAX_DIFF`) 이내인 가장 가까운 다른 주소를 hard negative로 채택
  - 입력/출력 파일을 인자로 받음: `python -m scripts.build_hard_negatives [입력_anchor_파일] [출력_파일]`
    (기본값: `data/address/STT발화_정답_학습데이터.txt` → `data/address/STT발화_정답_hard_neg.txt`)
  - 현재 데이터(486건) 기준 388건(79.8%)에서 hard negative 발견
- `src/dataset.py`에 `load_triplet_dataset(triplet_file)` 추가
  - `발화\t정답주소\t오답주소` 3열 탭 파일을 `{"anchor", "positive", "negative"}` Dataset으로 로드

## 아직 안 한 것 (다음에 이어서 할 일)

1. **실제 STT 2천 건 데이터 확보** (사용자가 진행 예정)
   - 확보되면 `발화\t정답주소` 탭 포맷으로 정리
2. **hard negative 재생성**: 새 2천 건 파일에 대해
   `python -m scripts.build_hard_negatives <새파일> <새파일>_hard_neg.txt` 실행
3. **`src/train.py` 수정 필요**
   - 현재는 `load_dataset(config.TRAIN_FAIL, config.TRAIN_ANSWER)`로 2열(fail/answer)만 로드
   - `load_triplet_dataset()`으로 만든 triplet 데이터셋을 함께 로드해서 합쳐야 함
     (예: `datasets.concatenate_datasets`로 anchor/positive만 있는 기존 데이터셋과
     anchor/positive/negative가 있는 triplet 데이터셋을 어떻게 합칠지 결정 필요 —
     column이 안 맞으므로 그냥 concat은 안 되고, negative 컬럼을 없는 행엔 None/빈값 처리하거나
     두 데이터셋을 별도 스텝으로 나눠 학습하는 방법도 고려)
4. **학습 데이터 구성 결정**
   - "실제 2천 건만" vs "2천 건 + 타겟 증강(번지수 위주)" 중 뭘로 갈지 확정
   - 확정되면 `scripts/augment_stt_answers.py`(현재 SAMPLES_PER_ADDRESS=25)를 참고해서
     번지수 변형 위주로 증강량 조절

## 관련 파일

- `scripts/build_hard_negatives.py`
- `src/dataset.py` (`load_triplet_dataset`)
- `data/address/STT발화_정답_hard_neg.txt` (현재 486건 기준 생성된 샘플, 참고용)
- `docs/eval_report_db.md` (이 작업의 근거가 된 평가 보고서)
