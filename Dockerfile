FROM python:latest

ARG project_dir=/gcn/
WORKDIR &project_dir

ADD main.py utils.py requirement.txt data $project_dir

RUN apt-get install gcc
RUN pip install -r requirement.txt

CMD ["python", "main.py"]
