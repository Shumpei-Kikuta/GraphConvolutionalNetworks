FROM python:latest

ARG project_dir=/gcn/
WORKDIR &project_dir

ADD . $project_dir

RUN apt-get install gcc
RUN pip install --upgrade pip
RUN pip install -r requirement.txt
RUN pip install scipy
RUN pip install torch
RUN pip install numpy

CMD ["python", "main.py"]