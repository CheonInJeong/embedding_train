# GPU 학습용 이미지. 반드시 저장소 루트를 build context로 빌드할 것:
#   docker build -f train/Dockerfile -t <registry>/<user>/bge-finetuning-train:latest .
#
# L4(Ada Lovelace, sm_89) 기준 CUDA 12.4 base 이미지 사용.
# 서버 NVIDIA 드라이버 버전이 이보다 낮으면(`nvidia-smi`로 확인) 태그를
# 더 낮은 CUDA 버전(예: cuda12.1-cudnn8-runtime)으로 바꿔야 한다.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

# torch는 base 이미지에 CUDA 빌드로 이미 설치되어 있으므로
# requirements.txt에는 포함하지 않는다. (pip로 다시 설치하면 CPU 빌드로
# 덮어써져 컨테이너 안에서 GPU를 못 쓰게 된다.)
COPY train/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/

ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface

# data/, models/ 는 이미지에 담지 않고 런타임에 volume으로 마운트한다.
# 실제 학습 로직(hard negative 포함)은 src/train.py에 있음 (루트의 train.py는
# 로컬 MPS/CPU용 구버전 스크립트라 여기서는 쓰지 않는다).
CMD ["python", "src/train.py"]
