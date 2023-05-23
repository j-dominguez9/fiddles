FROM docker.io/python:3.11
RUN mkdir /workspace
VOLUME /home/joaquin/Documents/SMU/ML2/HW/:/workspace:Z
WORKDIR /workspace

RUN apt update && apt upgrade -y && apt install curl -y 
RUN curl -sSL https://install.python-poetry.org | python3 -
RUN echo "export PATH=\"/root/.local/bin:$PATH\"">>/root/.bashrc
RUN /root/.local/bin/poetry config virtualenvs.in-project true
