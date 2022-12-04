import torch
import pytesseract
import numpy as np
import cv2
from time import time
import ocr

pytesseract.pytesseract.tesseract_cmd = r'c:\\users\\ilias\\appdata\\local\\programs\\python\\python39\scripts\\pytesseract.exe'

filename = 'Project\IMG_8716.mp4'
class plateRecognition:
    """
    Class implements Yolo5 model to make inferences on a  video using Opencv2.
    """

    def __init__(self,model_name):

       #self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
        self.path_to_video = "Project\IMG_8716.mp4"
        return cv2.VideoCapture(self.path_to_video)
        # self.path_to_video = "./test1.png"
        # return cv2.imread(self.path_to_video)



    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        if model_name:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_name, force_reload=True)
        else:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord



    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]

    def plot_boxes(self, results, frame, c):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]

        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0] * x_shape -5), int(row[1] * y_shape - 10), int(row[2] * x_shape + 5), int(
                    row[3] * y_shape + 10)
                bgr = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_TRIPLEX, 0.9, bgr, 2)
                cropped = frame[y1:y2, x1:x2]

                ocr_results = ocr.main(cropped)
                # cv2.imshow('ocr',ocr_results)
                # cv2.waitKey(0)

                ocr_results = cv2.copyMakeBorder(src=ocr_results,
                                                    top=0,
                                                    bottom=frame.shape[0] - ocr_results.shape[0],
                                                    left=0,
                                                    right=0,
                                                    borderType=cv2.BORDER_CONSTANT,
                                                    value=[0,0,0]
                                                    )
                frame = np.hstack((frame,ocr_results))
                # cv2.imshow(f'cropped{c}',cropped)
                # cv2.waitKey(0)
                # cv2.imwrite(f'cropped{c}.png',cropped)
                c += 1
        return frame,c


    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = self.get_video_capture()
        #assert cap.isOpened()
        c = 0
        while True:

            ret, frame = cap.read()
            assert ret

            #frame = cv2.resize(frame, (416, 416))

            start_time = time()
            results = self.score_frame(frame)

            frame, c = self.plot_boxes(results, frame, c)
            #text = pytesseract.image_to_string(frame)
            #print(text)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            # print(f"Frames Per Second : {fps}")

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 20), cv2.FONT_HERSHEY_TRIPLEX,0.5, (0, 0, 255), 1)


            cv2.imshow('Licence Plate Recognition', frame)


            if cv2.waitKey(33) & 0xFF == ord('\x1b'):
                break

        cap.release()


# Create a new object and execute.
detector = plateRecognition(model_name='Project\\best.pt')
detector()

