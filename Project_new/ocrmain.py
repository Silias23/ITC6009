import easyocr
import os
import datetime

class OCRRecognition:

    def __init__(self,path):
        self.reader = easyocr.Reader(['en'])
        self.path = path
        
    def ocr(self):
        self.results = self.reader.readtext(self.path)
        return self.results

    def checkResultFile(self):
        if os.path.isfile(self.path):
            return True
        else:
            #create the file
            file = open(self.path, "w")
            file.write("")
            file.close()
            return True

    def writeResults(self):
        if self.checkResultFile():
            for detection in self.results:
                if detection[2] > 0.45:
                    file = open(self.path, "a")
                    file.write(detection[1])
            try:
                file.write(' ' + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                file.write('\n')
                file.close()
            except:
                print('No license number characters were identified')

    def __call__(self):
        self.ocr()
        self.writeResults()


