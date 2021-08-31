FROM tensorflow/tensorflow:devel-gpu
WORKDIR /root

COPY requirements.txt ./requirements.txt
COPY trainer.py ./trainer.py
COPY raw_data/  ./raw_data/
COPY Makefile ./Makefile


RUN apt-get update 

# Ignore all of this for now we may need it later before bucket connection testing (remember to copy .json credentials if you do)
# RUN apt install curl
# RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
# RUN apt-get install apt-transport-https ca-certificates gnupg -y
# RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
# RUN apt-get update && apt-get install google-cloud-sdk -y
# RUN apt-get install google-cloud-sdk-app-engine-python -y

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["python", "trainer.py"]
