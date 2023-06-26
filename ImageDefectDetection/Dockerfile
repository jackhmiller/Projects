FROM python:3.9

ENV USER=docker
ENV HOME /home/$USER
RUN useradd -m $USER && echo $USER:$USER | chpasswd && adduser $USER sudo
RUN chown $USER:$USER $HOME

USER $USER
RUN mkdir -p $HOME/submission
WORKDIR $HOME/submission
COPY submission/ $HOME/submission/
COPY requirements.txt $HOME/
RUN echo building as $(whoami)
RUN echo $(which python)
ENV PATH=$HOME/.local/bin:$PATH

RUN /usr/local/bin/python -m pip install --disable-pip-version-check --upgrade pip
RUN pip install -r ../requirements.txt --disable-pip-version-check

CMD ["python3", "run_submission.py"]
