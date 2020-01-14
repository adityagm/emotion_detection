import cv2
import os
from pathlib import Path

def extractFrames(pathIn, pathOut, count=None):
    
 
    cap = cv2.VideoCapture(pathIn)
    while (cap.isOpened()):
 
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            print('Read %d frame: ' % count, ret)
            cv2.imwrite(os.path.join(pathOut, "frame{:d}.jpg".format(count)), frame)  # save frame as JPEG file
            count += 1
        else:
            break
 
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
 
def main():
    pathOut = 'F:\\Wenger\\data_all_vids'
    os.mkdir(pathOut)
    path = 'F:\Wenger'
    path_list = Path(path).glob('**\*.mp4')
    count = 0
    for files in path_list:
        extractFrames(str(files), pathOut, count)
 
if __name__=="__main__":
    main()
