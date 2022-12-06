import tracking
import ocr
import freq
import os
import ocrmain
import multiprocessing

def main():
    ocr.cleandir()
    track = tracking.plateRecognition(model_name='best.pt')
    track()

    # objs = [ocrmain.OCRRecognition(path=f'Results\TrackedPlates\Tracked{i}.jpg') for i in range(len(os.listdir('Results\TrackedPlates')))]
    # for obj in objs:
    #     obj()
        # rec = ocrmain.OCRRecognition(path=f'Results\TrackedPlates\Tracked{i}.jpg')
        # rec()
    ocr.findLicenses()
    plate = freq.find_frequency()
    print(plate)



if __name__ == '__main__':
    main()