FROM python:latest

ARG project_dir=/gcn_app/
WORKDIR $project_dir

ADD . $project_dir

RUN apt-get install gcc
RUN pip install --upgrade pip
RUN pip install -r requirement.txt

CMD ["python", "main.py"]