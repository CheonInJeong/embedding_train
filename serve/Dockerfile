FROM pytorch/torchserve:latest-cpu

USER root
WORKDIR /home/model-server

COPY serve/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY serve/handler.py handler.py
COPY models/bge-m3-address/config.json model_src/config.json
COPY models/bge-m3-address/model.safetensors model_src/model.safetensors
COPY models/bge-m3-address/tokenizer.json model_src/tokenizer.json
COPY models/bge-m3-address/tokenizer_config.json model_src/tokenizer_config.json

RUN torch-model-archiver \
    --model-name bge-m3-address \
    --version 1.0 \
    --handler handler.py \
    --serialized-file model_src/model.safetensors \
    --extra-files "model_src/config.json,model_src/tokenizer.json,model_src/tokenizer_config.json" \
    --export-path model-store

COPY serve/config.properties config.properties

EXPOSE 8080 8081 8082

CMD ["torchserve", "--start", "--foreground", "--ncs", \
     "--disable-token-auth", \
     "--model-store", "model-store", \
     "--models", "bge-m3-address=bge-m3-address.mar", \
     "--ts-config", "config.properties"]
