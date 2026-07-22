# GPU 서버에서 Docker로 학습하기

로컬 PC에서는 학습이 무리이므로, GPU(L4)가 있는 서버에서 Docker 컨테이너로 학습하고
결과 모델을 수동으로 HuggingFace Hub에 업로드하는 절차.

전체 흐름: **로컬에서 이미지 빌드 → 레지스트리에 push → GPU 서버에서 pull & 학습 → 학습 완료 후 `scripts/push_to_hub.py`로 수동 업로드**

## 0. 준비물

- 레지스트리 계정 (Docker Hub 또는 GHCR)
- GPU 서버에:
  - NVIDIA 드라이버 (L4 지원, `nvidia-smi`로 확인)
  - Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 설치 (`docker run --gpus all` 가 되어야 함)
- HuggingFace 계정 + write 권한 토큰 (https://huggingface.co/settings/tokens)

## 1. 로컬에서 이미지 빌드

`train/Dockerfile`은 `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` 베이스를 쓴다 (L4=sm_89 지원).
서버 드라이버가 이보다 오래됐으면 `nvidia-smi` 출력 상단의 CUDA 버전을 확인하고
Dockerfile 상단 주석대로 태그를 낮춰야 한다.

```bash
# 반드시 저장소 루트에서, context는 "."
docker build -f train/Dockerfile -t <registry>/<user>/bge-finetuning-train:latest .
```

`data/`, `models/`, `.venv/` 등은 `train/Dockerfile.dockerignore`(BuildKit 전용 ignore 파일이라
`serve/Dockerfile` 빌드에는 영향 없음)로 빌드 컨텍스트에서 제외되므로 이미지가 가볍다.
학습 데이터(`data/`)와 결과 모델(`models/`)은 이미지에 담기지 않고 런타임에 volume으로 마운트한다.

(Docker 23+ 는 기본으로 BuildKit을 쓰므로 별도 설정 불필요. 오래된 Docker라면
`DOCKER_BUILDKIT=1 docker build ...` 로 명시해야 `Dockerfile.dockerignore`가 적용된다.
적용이 안 돼도 빌드가 깨지진 않고 `models/`까지 컨텍스트로 올라가 느려지기만 한다.)

## 2. 레지스트리에 push

**Docker Hub**
```bash
docker login
docker push <user>/bge-finetuning-train:latest
```

**GHCR**
```bash
echo "$GITHUB_TOKEN" | docker login ghcr.io -u <github-user> --password-stdin
docker tag bge-finetuning-train:latest ghcr.io/<github-user>/bge-finetuning-train:latest
docker push ghcr.io/<github-user>/bge-finetuning-train:latest
```

## 3. GPU 서버에서 pull & 학습

서버에는 이미지를 실행할 때 필요한 최소한만 있으면 된다: 이 저장소의 `data/`(학습 데이터)와
`docker-compose.train.yml`, 그리고 결과를 받을 `models/` 디렉터리. (`git clone`으로 저장소 전체를
받아도 되고, `data/`만 rsync/scp해도 된다.)

```bash
docker pull <registry>/<user>/bge-finetuning-train:latest
docker tag <registry>/<user>/bge-finetuning-train:latest bge-finetuning-train:latest

# 저장소 루트에서
TRAIN_IMAGE=bge-finetuning-train:latest \
  docker compose -f docker-compose.train.yml run --rm train
```

또는 compose 없이 직접:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/workspace/data" \
  -v "$(pwd)/models:/workspace/models" \
  -v hf_cache:/workspace/.cache/huggingface \
  <registry>/<user>/bge-finetuning-train:latest
```

- `data/`, `models/`는 volume 마운트이므로 이미지를 다시 빌드하지 않아도 데이터/코드 변경분만
  반영하면 된다 (코드가 바뀌면 이미지는 다시 빌드해야 함).
- `hf_cache` 볼륨은 베이스 모델(`upskyy/bge-m3-korean`) HuggingFace 다운로드 캐시를 유지해서,
  컨테이너를 다시 띄울 때마다 재다운로드하지 않도록 한다.
- 학습 결과는 `src/config.py`의 `OUTPUT_DIR` (`models/bge-m3-address`)에 저장되며, 볼륨
  마운트 덕분에 서버의 `./models/bge-m3-address`에 그대로 남는다.

## 4. 학습 후 HuggingFace에 수동 업로드

같은 이미지로 커맨드만 바꿔서 실행 (모델을 서버 밖으로 옮길 필요 없음):

```bash
docker compose -f docker-compose.train.yml run --rm \
  -e HF_TOKEN=hf_xxx \
  -e HF_REPO_ID=your-name/bge-m3-address \
  train python -m scripts.push_to_hub
```

또는 `docker run`으로:

```bash
docker run --rm \
  -v "$(pwd)/models:/workspace/models" \
  -e HF_TOKEN=hf_xxx \
  -e HF_REPO_ID=your-name/bge-m3-address \
  bge-finetuning-train:latest \
  python scripts/push_to_hub.py
```

옵션은 `--repo-id`/`--token`/`--private`로도 넘길 수 있다 (`scripts/push_to_hub.py --help`).
HF_TOKEN은 절대 이미지에 굽거나 커밋하지 말고 매번 `-e`/`--env-file`로만 전달할 것.

## 참고

- `train/Dockerfile` — 학습용 이미지 (CUDA base, torch는 base 이미지 그대로 사용)
- `train/requirements.txt` — 학습에 필요한 python 패키지 (torch 제외)
- `docker-compose.train.yml` — GPU 서버용 compose (기존 `docker-compose.yml`은 로컬 postgres/torchserve 추론용이라 별개)
- `scripts/push_to_hub.py` — 학습된 모델을 HF Hub에 업로드하는 수동 스크립트
