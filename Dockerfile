FROM python:3.8

WORKDIR /app

COPY graph_embedding/ .

RUN pip install -r requirements.txt

CMD ["/bin/bash"]