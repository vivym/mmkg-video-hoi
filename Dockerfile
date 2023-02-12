FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /work

RUN pip install --no-cache-dir --upgrade mmkg-mm-entity-linking

CMD ["uvicorn", "mmkg_mm_entity_linking:app", "--host", "0.0.0.0", "--port", "80", "--root-path", "/mmkg-mm-entity-linking"]
