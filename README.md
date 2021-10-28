# DOCKER based FASTAPI server that reads uploaded videos and returns found objects

## Pre-install
1. Go to ./services/AI/: `cd services/AI/`
2. Change install.sh to 777 chmod: `sudo chmod 777 install.sh`
3. Run it: `./install.sh`

## Install

1. Add to your docker-compose file next element:
```
ai:
    restart: always
    build:
        context: ./services/AI
    command: gunicorn main:app --bind 0.0.0.0:4999 -w 4 -k uvicorn.workers.UvicornWorker
    volumes:
      - ./services/AI/:/home/app/
```

2. Then create client.py:

```python=
#!/usr/bin/python3

import datetime
import sys
import requests
import os

url = "http://ai:4999/uploadfile/" # For docker container
url = "http://0.0.0.0:4999/uploadfile/" # For external client

if __name__ == "__main__":
    files = open(PATHTOVIDEOFILE, 'rb')

    r = requests.post(url, files={"file": files})

    print(r.content)
```

3. Run and get output:

```
[{"object":"car","confidence":0.9878160357475281}]
```