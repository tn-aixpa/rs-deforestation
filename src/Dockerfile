FROM ghcr.io/tn-aixpa/rsde-tools:2.0

USER 1000

RUN conda init bash && . ~/.bashrc 
#RUN echo "conda activate rsde" >> ~/.bashrc

WORKDIR /app
COPY main.py .
COPY requirements.txt .
RUN mkdir /app/utils
COPY utils /app/utils
RUN mkdir /app/data
RUN mkdir /app/output
RUN mkdir /app/bosco
#COPY bosco /app/bosco
#COPY data /app/data

RUN python -m pip install -r requirements.txt

#ENTRYPOINT [ "python","main.py" ]
ENTRYPOINT [ "/bin/bash" ]