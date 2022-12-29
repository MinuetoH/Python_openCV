# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 09:23:34 2022

@author: kimyh
"""

import cv2

#이미지 형태 분석
def print_matInfo(name, image):
    #image : 이미지를 읽은 배열값. 이미지데이터
    #image.dtype : 배열 요소의 자료형.
    if image.dtype == 'uint8':     mat_type = "CV_8U" #부호없는8비트(0~255)
    elif image.dtype == 'int8':    mat_type = "CV_8S" #부호있는8비트(-128~127)
    elif image.dtype == 'uint16':  mat_type = "CV_16U"#부호없는16비트
    elif image.dtype == 'int16':   mat_type = "CV_16S"#부호있는16비트
    elif image.dtype == 'float32': mat_type = "CV_32F"#부호있는32비트 실수형
    elif image.dtype == 'float64': mat_type = "CV_64F"#부호있는64비트 실수형
    #image.ndim : 배열의 차수 
    nchannel = 3 if image.ndim == 3 else 1  
    print("%12s: dtype(%s), channels(%s) -> mat_type(%sC%d)"
          % (name, image.dtype, nchannel, mat_type,  nchannel))
#imread : 이미지파일을 배열 저장
gray2gray = cv2.imread("images/read_gray.jpg", cv2.IMREAD_GRAYSCALE)
gray2color = cv2.imread("images/read_gray.jpg", cv2.IMREAD_COLOR)
gray2colora = cv2.imread("images/read_gray.jpg", cv2.IMREAD_UNCHANGED)
color2gray = cv2.imread("images/read_color.jpg", cv2.IMREAD_GRAYSCALE)
color2color = cv2.imread("images/read_color.jpg", cv2.IMREAD_COLOR)
color2colora = cv2.imread("images/read_color.jpg", cv2.IMREAD_UNCHANGED)
print_matInfo("gray2gray", gray2gray)
print_matInfo("gray2color", gray2color)
print_matInfo("gray2colora", gray2colora)
print_matInfo("color2gray", color2gray)
print_matInfo("color2color", color2color)
print_matInfo("color2colora", color2colora)

color1 = cv2.imread("images/read_16.tif", cv2.IMREAD_UNCHANGED)
color2 = cv2.imread("images/read_32.tif", cv2.IMREAD_UNCHANGED)
color1.shape
color2.shape
color1[10,10]
color2[10,10]
cv2.imshow("read_16.tif",color1)
cv2.imshow("read_32.tif",color2)
print_matInfo("color1", color1)
print_matInfo("color2", color2)

#사진의 밝기 조정
image = cv2.imread("images/read_gray.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("origimal image",image)
#OpenCv함수를 이용하여 사진의 밝기 조정
dst1 = cv2.add(image,100) #image 배열의 값+100
dst2 = cv2.subtract(image,100) #image 배열의 값-100
cv2.imshow("dst1 image",dst1)
cv2.imshow("dst2 image",dst2)
#numpy 배열을 이용하여 사진의 밝기 조정
dst3=image+100 #image 배열의 값+100
dst4=image-100 #image 배열의 값-100
cv2.imshow("dst3 image",dst3)
cv2.imshow("dst4 image",dst4)

#### 동영상 파일
import cv2
capture = cv2.VideoCapture(0) #카메라 객체 연결
if capture.isOpened() == False: 
    raise Exception("카메라 연결 안됨")
#카메라 속성값    
print("너비 %d" % capture.get(cv2.CAP_PROP_FRAME_WIDTH)) #가로길이
print("높이 %d" % capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) #세로길이
print("노출 %d" % capture.get(cv2.CAP_PROP_EXPOSURE))
print("밝기 %d" % capture.get(cv2.CAP_PROP_BRIGHTNESS)) 
#동영상에 지정된 위치 문자 출력 함수
#color=(120, 200, 90) : BGR
def put_string(frame, text, pt, value, color=(120, 200, 90)):  
    #frame : 동영상 출력 영역. 이미지.
    #text : 출력할 문자내용
    #pt : 문자 출력 위치. (10, 40)
    text += str(value)  #EXPOS: -1 , 출력할 문자내용
    shade = (pt[0] + 2, pt[1] + 2) #(12,42) 그림자의 위치값
    font = cv2.FONT_HERSHEY_SIMPLEX  #폰트설정.(영문만가능)
    #0.7 : fontscale. 폰트 크기
    #(0, 0, 0) : 검정색
    # 2:두께
    cv2.putText(frame, text, shade, font, 0.7, (0, 0, 0), 2) #그림자효과
    cv2.putText(frame, text, pt, font, 0.7, color, 2)

#카메라 영상 출력
while True:  #무한 반복.
    #frame : 순간 이미지
    ret, frame = capture.read() #카메라영상 받기
    if not ret: break 
    if cv2.waitKey(30) >= 0: break #30 스페이스바 입력시 종료
    exposure = capture.get(cv2.CAP_PROP_EXPOSURE) #노출값
    put_string(frame, "EXPOS: ", (10, 40), exposure)
    #frame : 텍스트를 출력한 이미지 파일
    title = "View Frame from Camera"
    cv2.imshow(title , frame) #이미지 화면 출력
capture.release()  #카메라 접속 종료

#동영상을 파일로 저장하기
import cv2
capture = cv2.VideoCapture(0)
if capture.isOpened() == False: 
    raise Exception("카메라 연결 안됨")
fps=20.0 #초당 프레임 수
delay=round(1000/fps)
size=(640,480)
# *"DX50" : 코덱 종류 설정.
fourcc=cv2.VideoWriter_fourcc(*"DX50") #코덱설정

print("프레임 해상도:",size)
print("압축코덱숫자:",fourcc) 
print("delay: %2d ms" % delay)
print("fps: %.2f" % fps)
#카메라 설정
capture.set(cv2.CAP_PROP_ZOOM,1) 
capture.set(cv2.CAP_PROP_FOCUS,0) 
capture.set(cv2.CAP_PROP_FRAME_WIDTH,size[0])
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,size[1])
#동영상 출력파일 설정
writer = cv2.VideoWriter\
    ("images/video_file.avi",fourcc,fps,size)
if writer.isOpened()==False : 
       raise Exception("동영상 파일 개방 오류")
while True:
    ret,frame = capture.read()  #순간이미지 파일
    if not ret : break
    if cv2.waitKey(30) >= 0: break
    writer.write(frame) #이미지 저장.
    cv2.imshow("View Frame from Camera",frame) #현재이미지출력
writer.release() #출력 접속 종료
capture.release() #카메라 접속 종료

#저장된 동영상 파일을 출력하기
capture = cv2.VideoCapture("images/video_file.avi")
if not capture.isOpened() : 
    raise Exception("동영상 파일 개방 오류")
frame_rate = capture.get(cv2.CAP_PROP_FPS) #초당 frame 수
print("frame_rate:",frame_rate) #20   

while True:
    ret,frame = capture.read()
    if not ret : break  #동영상파일 읽기 종료
    if cv2.waitKey(30) >= 0: break
    cv2.imshow("View Frame from Camera",frame) 
capture.release()  #동영상 파일 접속 종료   

#색상 수정하기
capture = cv2.VideoCapture("images/video_file.avi")
if not capture.isOpened() : 
    raise Exception("동영상 파일 개방 오류")
frame_cnt = 0                 
while True:
	ret, frame = capture.read() 
	if not ret or cv2.waitKey(50) >= 0: break 
    #cv2.split(frame) : 
	blue, green, red = cv2.split(frame) #bgr 색상의 배열을 분리
	frame_cnt += 1
    #cv2.add(blue, 100, blue) : blue+100 배열값 => blue 
	if 50 <= frame_cnt < 100 : cv2.add(blue, 100, blue)
	elif 100 <= frame_cnt < 150: cv2.add(green, 100, green)
	elif 150 <= frame_cnt < 300: cv2.add(red  , 100, red)   
    #cv2.merge( [blue, green, red] ) : bgr 배열을 병합.
	frame = cv2.merge( [blue, green, red] )
	put_string(frame, "frame_cnt : ", (20, 320), frame_cnt)
	cv2.imshow("Read Video File", frame)
capture.release()

#이미지 색상으로 분리하기
import cv2
import numpy as np

image = cv2.imread("images/read_color.jpg", cv2.IMREAD_COLOR)
cv2.imshow("image", image)
bgr = cv2.split(image) 
zero = np.zeros_like(image,dtype="uint8")
z = cv2.split(zero)
blue = cv2.merge([bgr[0],z[1],z[2]])
green = cv2.merge([z[0],bgr[1],z[2]])
red = cv2.merge([z[0],z[1],bgr[2]])
cv2.imshow("Blue channel" , blue)         # blue 채널
cv2.imshow("Green channel", green)         # green 채널
cv2.imshow("Red channel"  , red)         # red 채널


################
# 숫자 인식하기
################
import cv2
import numpy as np
import pickle,gzip,os

from urllib.request import urlretrieve
import matplotlib.pyplot as plt

def load_mnist(filename) :
    if not os.path.exists(filename) : #존재하지 않으면
        link = \
"https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"
        urlretrieve(link,filename) #link에 전달된 파일을 filename으로 저장
        with gzip.open(filename,"rb") as f: #압축파일 읽기
            return pickle.load(f,encoding="latin1")

train_set,valid_set,test_set=load_mnist("mnist.pkl.gz")
#테스트데이터 : 훈련 종료 후평가를 위한 데이터
#검증데이터 : 훈련도중 평가를 위한 데이터
train_data,train_label = train_set  #훈련데이터
test_data,test_label=test_set       #테스트데이터
valid_data, valid_label = valid_set #검증데이터
print("train_data[0]=",train_data[0])
print("train_label[0]=",train_label[0])
train_data.shape # (50000, 784) 50000개의 행. 숫자이미지 행.
#                    784 : 28*28 => 1차원배열로 생성.
#                    50000개의 숫자 이미지값.
train_label.shape  #50000. 정답.
test_data.shape    #(10000, 784)
valid_data.shape   #(10000, 784)
#이미지 출력하기
def graph_image(data, lable, title, nsample):
    plt.figure(num=title, figsize=(6, 9))
    #rand_idx : 0~49999 까지의 수중 24개의 데이터 저장
    rand_idx = np.random.choice(range(data.shape[0]), nsample)
    print(rand_idx)
    for i, id in enumerate(rand_idx):
        #data[id] : 한개 행. 784개=> 28*28
        img = data[id].reshape(28, 28) #2차원배열로 변경
        plt.subplot(6, 4, i + 1) #6행4열 순서. 24개의 이미지 출력
        plt.axis('off') #축을 안보이도록 설정
        plt.imshow(img, cmap='gray') #이미지 출력
        plt.title('%s: %d' % (title , lable[id]))
    plt.tight_layout()
    
graph_image(train_data, train_label, 'label', 24)   
#학습하기
knn = cv2.ml.KNearest_create() #KNN 알고리즘.
#훈련하기
#train_data : 훈련데이터. 50000개.
#cv2.ml.ROW_SAMPLE : 행값이 데이터. 1개 행이 학습데이터 1개.
#train_label : 정답.
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_label)
#예측하기.
#test_data[:100] : 100개만 예측하기.
#k=5 : KNN 알고리즘 근접한 5개의 점을 선택.
_, resp, _ , _ = knn.findNearest(test_data[:100], k=5)
#resp.flatten() : 1차원배열로 변경. [[값]]
accur = sum(test_label[:100] == resp.flatten()) / len(resp)
print("정확도=", accur*100, '%')
graph_image(test_data[:100], resp, 'predict', 24)
  








