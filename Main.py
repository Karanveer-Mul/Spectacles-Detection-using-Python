import datetime
import pytz
import cv2
import numpy as np
import pandas as pd
import math
import time
import datetime
import warnings
from mlxtend.image import extract_face_landmarks


pic_info = {'Year':[],'Month':[],'Day':[],'Hour':[],'Minute':[],'Second':[],'Microsecond':[],'Timezone':[],'isWearingSpecs':''}
df = pd.DataFrame(pic_info,columns=['Year','Month','Day','Hour','Minute','Second','Microsecond','Timezone','isWearingSpecs'])

#If we want to append data instead of creating new datasets each time
#df = pd.read_csv("pic_info.csv")

warnings.filterwarnings('error','No face detected.')
url = "http://192.168.43.1:8080/video"
cap = cv2.VideoCapture(0)
wi = 1
ni = 1

while True:    
    return_value,frame = cap.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    process_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
    cv2.imshow('test',small_frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    img = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
    try:
        landmarks = extract_face_landmarks(img)
        
        #Individual facial features
        left_outer_eye = landmarks[36]
        right_outer_eye = landmarks[45]

        #We will use trigonometry to find roatation angle required to normalize the face
        h = left_outer_eye[1]-right_outer_eye[1] #Height of the triangle
        w = left_outer_eye[0]-right_outer_eye[0] #Width of the triangle
        angle = math.degrees(math.atan(h/w)) #Finding tan inverse and converting to degress for angle of rotation
        image_center = tuple(np.array(img.shape[1::-1])/2) 
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0) 
        rotated_img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        landmarks = extract_face_landmarks(rotated_img)

    except UserWarning:
        continue 
    img = rotated_img[landmarks[24][1]:landmarks[28][1],landmarks[36][0]:landmarks[45][0]] #Area to crop

    edges = cv2.Canny(img,100,100)
    h,w = edges.shape

    #Extracting coordinates from edges
    indices = np.where(edges != [0])
    coordinates =  list(zip(indices[1], indices[0]))

    #Declaring iterator and list
    i = 0
    count = []

    for x in range(int(w*0.45),int(w*0.55)):
        count.append(0)
        for y in range(0,h):
            if (x,y) in coordinates:
                count[i] += 1
        i += 1
    #print(count)
    flag = 1
    
    cur_time = datetime.datetime.now(tz=pytz.timezone('Asia/Kolkata'))
    for x in count:
        if x<2:
            flag = 0
            print("Spectacles not Detected\n")
            df = df.append({'Year':cur_time.year,'Month':cur_time.month,'Day':cur_time.day,'Hour':cur_time.hour,
            'Minute':cur_time.minute,'Second':cur_time.second,'Microsecond':cur_time.microsecond,'Timezone':cur_time.tzinfo,
            'isWearingSpecs':'No'}, ignore_index = True)
            if(ni==21):
                ni=1
            cv2.imwrite(filename = 'NotWearing\ '+str(ni)+'.jpg', img = frame)
            ni+=1
            break
    if (flag==1):
        print("The image has spectacles\n")
        df = df.append({'Year':cur_time.year,'Month':cur_time.month,'Day':cur_time.day,'Hour':cur_time.hour,
        'Minute':cur_time.minute,'Second':cur_time.second,'Microsecond':cur_time.microsecond,'Timezone':cur_time.tzinfo,
        'isWearingSpecs':'Yes'}, ignore_index = True)
        if(wi==21):
            wi=1
        cv2.imwrite(filename = 'Wearing\ '+str(wi)+'.jpg', img = frame)
        wi+=1
    #time.sleep(1)
df.to_csv(r'picture_info.csv', index = False, header = True)
print(df)
cap.release()
cv2.destroyAllWindows()
