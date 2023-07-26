FROM python:3.9

WORKDIR /code

RUN pip install \
    torch==1.12.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    && rm -rf /root/.cache/pip

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./server.py ./image_embeds.npy /code/

CMD ["gunicorn", "server:app", "--workers", "2", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:80"]