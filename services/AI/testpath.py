#!/usr/bin/python3

import datetime
import sys
import requests
import os

url = "http://0.0.0.0:4999/uploadfile/"

if __name__ == "__main__":
    files = open("video/1.mp4", 'rb')

    r = requests.post(url, files={"file": files})

    print(r.content)
    
