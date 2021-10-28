import numpy as np
import time
import cv2
import logging
from fastapi import FastAPI, File, UploadFile
import datetime
import uvicorn
import aiofiles
import os

logging.basicConfig(format='%(asctime)s| %(levelname)s: %(message)s', level=logging.DEBUG)
logging.getLogger("urllib3").setLevel(level=logging.WARNING)

class AI:
    def __init__(self) -> None:
        self.INPUT_FILE='img/test.jpg'
        self.OUTPUT_FILE='predicted.jpg'
        self.LABELS_FILE='data/coco.names'
        self.CONFIG_FILE='cfg/yolov3.cfg'
        self.WEIGHTS_FILE='weights/yolov3.weights'
        self.CONFIDENCE_THRESHOLD=0.3
        self.LABELS = open(self.LABELS_FILE).read().strip().split("\n")
        np.random.seed(4)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
            dtype="uint8")
        self.net = cv2.dnn.readNetFromDarknet(self.CONFIG_FILE, self.WEIGHTS_FILE)
        self.boxes = []
        self.confidences = []
        self.classIDs = []

    def analyze_frame(self, filename):
        self.image = cv2.imread(filename)
        (H, W) = self.image.shape[:2]
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        blob = cv2.dnn.blobFromImage(self.image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(ln)
        end = time.time()
        logging.debug("[INFO] YOLO took {:.6f} seconds".format(end - start))
        self.boxes = []
        self.confidences = []
        self.classIDs = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                self.confidence = scores[classID]
                if self.confidence > self.CONFIDENCE_THRESHOLD:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    self.boxes.append([x, y, int(width), int(height)])
                    self.confidences.append(float(self.confidence))
                    self.classIDs.append(classID)
        self.idxs = cv2.dnn.NMSBoxes(self.boxes, self.confidences, self.CONFIDENCE_THRESHOLD,
            self.CONFIDENCE_THRESHOLD)
        logging.debug(self.classIDs)
        outputs = []
        state = False
        for i in self.idxs.flatten():
            id = self.classIDs[i]
            if self.LABELS[id] in ["truck", "person", "car", "boat"]:
                outputs.append({"object": self.LABELS[self.classIDs[i]], "confidence": self.confidences[i]})
        if len(outputs) > 0:
            # self.draw_image()
            state = True
        return state, outputs
    
    def analyze_video(self, filename):
        vidcap = cv2.VideoCapture(filename)
        success,image = vidcap.read()
        count = 0
        numFrameToSave = 3
        while success:
            success,image = vidcap.read()
            if not success:
                break
            if (count % numFrameToSave ==0):
                cv2.imwrite("img/img.jpg", image)
                state, output = self.analyze_frame("img/img.jpg")
                if state:
                    return output
            if cv2.waitKey(10) == 27:                     
                break
            count += 1
        return []

    def draw_image(self):
        if len(self.idxs) > 0:
            for i in self.idxs.flatten():
                (x, y) = (self.boxes[i][0], self.boxes[i][1])
                (w, h) = (self.boxes[i][2], self.boxes[i][3])
                color = [int(c) for c in self.COLORS[self.classIDs[i]]]
                cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.LABELS[self.classIDs[i]], self.confidences[i])
                cv2.putText(self.image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
        cv2.imwrite("output.png", self.image)


app = FastAPI()
ai = AI()


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    file_location = f"video/anal{file.filename}"
    async with aiofiles.open(file_location, 'wb+') as out_file:
        content = await file.read()
        await out_file.write(content)
    out = ai.analyze_video(file_location)
    os.remove(f"video/anal{file.filename}")
    return out


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=False, workers=4)