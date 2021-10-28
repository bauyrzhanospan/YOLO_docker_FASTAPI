#!/usr/bin/python3

import datetime
import sys
import requests
import os

url = "http://0.0.0.0:5000/uploadfile/"

robotID = "1"

if __name__ == "__main__":
    files = open("video/1.mp4", 'rb')

    r = requests.post(url, files={"file": files})

    print(r.content)
    
