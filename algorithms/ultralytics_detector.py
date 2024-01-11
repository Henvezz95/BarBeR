from ultralytics import YOLO


class YOLO_detector:
    def __init__(self, model_path, imgsz):
        self.model = YOLO(model_path, task='detect')
        self.imgsz = imgsz

    def detect(self, img):
        detection = self.model(img.astype('uint8'), verbose=False, imgsz=self.imgsz)
        names = detection[0].names
        result = []
        predictions = []
        confidences = []
        boxes = detection[0].boxes.data.cpu().numpy()
        for box in (boxes):
            x0, y0, x1, y1, conf, pred = box
            result.append([x0,y0, x1-x0, y1-y0])
            predictions.append(names[int(pred)])
            confidences.append(conf)

        return result, predictions, confidences #Boxes, classes, confidence scores