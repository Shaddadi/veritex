FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
RUN pip install --upgrade pip

WORKDIR /art
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
CMD ["bash"]
