FROM amazon/aws-lambda-python:3.9

RUN python3.9 -m pip install --upgrade pip

COPY app/ /var/task/

RUN pip install awslambdaric
RUN pip install -r /var/task/requirements.txt

ENTRYPOINT [ "python3.9", "-m", "awslambdaric" ]
CMD ["main.handler"]
