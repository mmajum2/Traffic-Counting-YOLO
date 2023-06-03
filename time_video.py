import cv2
import os
import  openpyxl
def write_to_excel(time_count,duration_mins,entry,exit,filepath='./output.xlsx'):

    if (os.path.exists(filepath)):
        workbook_obj = openpyxl.load_workbook(filepath)
        sheet_obj = workbook_obj.active
        start_time = (time_count - 1) * duration_mins
        finish_time = (time_count - 1) * duration_mins + duration_mins
        # '%02d:%02d' % start_time,1
        col1 = '%02d:%02d' % (start_time,1)
        col2 = '%02d:%02d' % (finish_time, 0)
        print(col1)
        print(col2)
        col3 = entry
        col4 = exit
        sheet_obj.append([col1, col2, col3, col4])
    else:
        workbook_obj = openpyxl.Workbook()
        sheet_obj = workbook_obj.active
        col1 = 'Time Start'
        col2 = 'Time End'
        col3 = 'Entry'
        col4 = 'Exit'
        sheet_obj.append([col1, col2, col3, col4])

        start_time = (time_count - 1) * duration_mins
        finish_time = (time_count - 1) * duration_mins + duration_mins

        col1 = '%02d:%02d' % (start_time,1)
        col2 = '%02d:%02d' % (finish_time, 0)
        # print(col1)
        # print(col2)
        col3 = entry
        col4 = exit
        sheet_obj.append([col1, col2, col3, col4])

    workbook_obj.save(filepath)



cap = cv2.VideoCapture("./path2your_video/2- 5min.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps

print('fps = ' + str(fps))
print('number of frames = ' + str(frame_count))
print('duration (S) = ' + str(duration))
minutes = int(duration/60)
seconds = duration%60
print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

cap.release()
write_to_excel(time_count=1,duration_mins=5,entry=2,exit=1)
# print("{0:0=2d}".format(1))
