FROM jjanzic/docker-python3-opencv

ENV WORKDIR=/usr/src/app
ENV USER=app
ENV APP_HOME=/home/app/
# ENV . /app
# WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR $WORKDIR

# RUN pip3 install --upgrade pip
COPY ./requirements.txt $WORKDIR/requirements.txt
RUN pip3 install -r requirements.txt
RUN chmod 777 install.sh
RUN ./install.sh

RUN adduser --system --group $USER
# RUN mkdir $APP_HOME
WORKDIR $APP_HOME

ADD . $APP_HOME
RUN chown -R $USER:$USER $APP_HOME
USER $USER