import tracking
import ocr
import freq
import easyocr
import os

def main():
    #reset the results folder
    ocr.cleandir()
    #create the ocr reader object that will be used later
    reader = easyocr.Reader(['en'])
    #create an instance of custom plate recognition tracking that uses custom trained yolov5 model to identify the location of the license plate on each frame
    track = tracking.plateRecognition('best.pt')
    track()
    #run the processed pictures through the ocr reader
    ocr.findLicenses(reader)
    plate = freq.find_frequency()
    print(plate)



if __name__ == '__main__':
    main()