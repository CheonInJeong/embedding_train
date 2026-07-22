"""학습이 끝난 SentenceTransformer 모델을 HuggingFace Hub에 업로드.

사용 예:
    python scripts/push_to_hub.py --repo-id your-name/bge-m3-address --token hf_xxx
    # 또는 환경변수로
    HF_TOKEN=hf_xxx HF_REPO_ID=your-name/bge-m3-address python scripts/push_to_hub.py
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentence_transformers import SentenceTransformer

from src import config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID"), help="예: your-name/bge-m3-address")
    parser.add_argument("--model-dir", default=config.OUTPUT_DIR, help="업로드할 로컬 모델 경로 (기본: 학습 output 경로)")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument(
        "--private",
        action="store_true",
        default=os.environ.get("HF_PRIVATE", "").lower() in ("1", "true"),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.repo_id:
        sys.exit("--repo-id 또는 HF_REPO_ID 환경변수가 필요합니다.")
    if not args.token:
        sys.exit("--token 또는 HF_TOKEN 환경변수가 필요합니다.")
    if not os.path.isdir(args.model_dir):
        sys.exit(f"모델 디렉터리를 찾을 수 없습니다: {args.model_dir}")

    print(f"Loading model from {args.model_dir} ...")
    model = SentenceTransformer(args.model_dir)

    print(f"Pushing to hub: {args.repo_id} (private={args.private}) ...")
    model.push_to_hub(
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )

    print(f"Done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
