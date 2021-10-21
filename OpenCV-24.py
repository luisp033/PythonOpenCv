import cv2
import time
import face_recognition as FR

font=cv2.FONT_HERSHEY_SIMPLEX
width = 640
height = 360
new_frame_time = 0
prev_frame_time = 0

cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))


donFace=FR.load_image_file('C:/Users/Utilizador/Documents/Python/pyAI3.6/demoImages/known/Donald Trump.jpg')
faceLoc = FR.face_locations(donFace)[0]
donFaceEncode=FR.face_encodings(donFace)[0]

nancyFace=FR.load_image_file('C:/Users/Utilizador/Documents/Python/pyAI3.6/demoImages/known/Nancy Pelosi.jpg')
faceLoc = FR.face_locations(nancyFace)[0]
nancyFaceEncode=FR.face_encodings(nancyFace)[0]

LuisFace=FR.load_image_file('C:/Users/Utilizador/Documents/Python/pyAI3.6/demoImages/known/Luis.jpg')
faceLoc = FR.face_locations(LuisFace)[0]
LuisFaceEncode=FR.face_encodings(LuisFace)[0]

knownEncodings=[donFaceEncode,nancyFaceEncode,LuisFaceEncode]
names=['Trump','Pelosi','Luis']

while True:
    ignore, frame =  cam.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    unknownFace=frame
    unknownFaceBGR=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    faceLocations=FR.face_locations(unknownFace)
    unknownEncodings=FR.face_encodings(unknownFace,faceLocations)

    for faceLocation,unknownEncoding in zip(faceLocations,unknownEncodings):
        top,right,bottom,left = faceLocation
        print(faceLocation)
        cv2.rectangle(unknownFaceBGR,(left,top),(right,bottom),(255,0,0),3)
        name='Unknown'
        matches=FR.compare_faces(knownEncodings,unknownEncoding)
        print(matches)
        if True in matches:
            matchIndex=matches.index(True)
            #print(matchIndex)
            print(names[matchIndex])
            name= names[matchIndex]
        cv2.putText(unknownFaceBGR,name,(left,top),font,.75,(0,0,255),2)

    # Write fps in left corner
    new_frame_time = time.time()    
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(unknownFaceBGR, fps, (7, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA) 

    myFrameRGB=cv2.cvtColor(unknownFaceBGR,cv2.COLOR_BGR2RGB)

    cv2.imshow('Myfaces',myFrameRGB)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cam.release()