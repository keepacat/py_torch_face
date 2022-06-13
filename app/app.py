import os

# ffmpeg -ss 0:0:12 -i xxx.mp4 -r 1 -f image2 yyy_%03d.jpg -y
# str = 'C:/ProgramData/Anaconda3/pkgs/ffmpeg-4.3.1-ha925a31_0/Library/bin/ffmpeg.exe -ss 0:0:0 -i ./video.flv -r 1 -f image2 ./images/%05d.jpg -y'
# os.popen(str)

import face_recognition
from PIL import Image
import cv2

path_images = "./images/"
path_person = "./person/"
path_face = "./face/"
path_face2 = "./face4/"

def face():
    #存储知道人名列表
    known_names=[] 
    #存储知道的特征值
    known_encodings=[]
    for image_name in os.listdir(path_person):
        face_image = face_recognition.load_image_file(path_person+image_name) #加载图片
        face_locations = face_recognition.face_locations(face_image)#获得所有人脸位置
        image_face_encoding = face_recognition.face_encodings(face_image, face_locations)[0] #获得128维特征值
        known_names.append(image_name.split(".")[0])
        known_encodings.append(image_face_encoding)
    print(known_encodings)

    name_index = 0
    for image_name in os.listdir(path_images):
        face_image = face_recognition.load_image_file(path_images+image_name) #加载图片
        face_locations = face_recognition.face_locations(face_image)#获得所有人脸位置
        face_encodings = face_recognition.face_encodings(face_image, face_locations) #获得128维特征值
        for index, face_encoding in enumerate(face_encodings):
            image = Image.open(path_images+image_name)
            (top, right, bottom, left) = face_locations[index]
            img = image.crop((left, top, right, bottom))

            matches = face_recognition.compare_faces(known_encodings, face_encoding,tolerance=0.2)
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index] + "_" + str(name_index) + ".jpg"
                img.save(path_face2 + name)
            else:
                name = "unknown_" + str(name_index) + ".jpg"
                img.save(path_face + name)
            name_index += 1
            

    
def remove():
    #打开摄像头，0表示内置摄像头
    video_capture = cv2.VideoCapture(0) 
    process_this_frame = True
    while True:
        ret, frame = video_capture.read()
        # opencv的图像是BGR格式的，而我们需要是的RGB格式的，因此需要进行一个转换。
        rgb_frame = frame[:, :, ::-1]
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_frame)#获得所有人脸位置
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations) #获得人脸特征值
            face_names = [] #存储出现在画面中人脸的名字
            for face_encoding in face_encodings:         
                matches = face_recognition.compare_faces(known_encodings, face_encoding,tolerance=0.5)
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_names[first_match_index]
                else:
                    name="unknown"
                face_names.append(name)

        process_this_frame = not process_this_frame

        # 将捕捉到的人脸显示出来
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2) # 画人脸矩形框
            # 加上人名标签
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX 
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


face()