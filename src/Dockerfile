FROM ghcr.io/tn-aixpa/rsde-tools:2.6


RUN conda init bash && . ~/.bashrc 

WORKDIR /app
COPY main.py .
COPY requirements.txt .
RUN mkdir /app/utils
COPY utils /app/utils
RUN mkdir /app/data
RUN mkdir /app/output
RUN mkdir /app/bosco

RUN python -m pip install -r requirements.txt

ENTRYPOINT [ "/bin/bash" ]