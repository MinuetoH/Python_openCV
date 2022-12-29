# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 11:42:32 2022

@author: kimyh
20221226.py
openCV 예제 : 이미지 처리를 위한 툴
    pip install opencv-python
    
    빅데이터 조건(3V)
    1. Volumn : 대용량
    2. Velocity : 속도. 처리속도가 빠르다.
    3. Variety : 데이터의 다양성
        - 정형데이터 : dbms, csv, 엑셀 => pandas, numpy, ...
        - 반정형데이터 : xml, html, json => BeautifulSoup, solenium, json
        - 비정형데이터 : 이미지, 동영상 => opencv
"""

import cv2
title1,title2,title3 = "gray2gary", "gray2color", "gray2colora"
# imread : 이미지 파일을 읽기. 행렬 데이터로 변환.
# cv2.IMREAD_GRAYSCALE : 흑백이미지로 처리
# cv2.IMREAD_COLOR : 컬러이미지로 처리
# cv2.IMREAD_UNCHANGED : 원래이미지로 처
gray2gary = cv2.imread("images/read_gray.jpg", cv2.IMREAD_GRAYSCALE)
gray2color = cv2.imread("images/read_gray.jpg", cv2.IMREAD_COLOR)
gray2colora = cv2.imread("images/read_gray.jpg", cv2.IMREAD_UNCHANGED)
if (gray2gary is None or gray2color is None) :
    raise Exception("영상파일 읽기 에러")
type(gray2gary)
gray2gary.shape  # (300, 400)
gray2gary.ndim   # 2
gray2gary[0]
gray2color.shape #(300, 400, 3)
gray2color.ndim  # 3
gray2color[0]
gray2colora.shape #(300, 400, 3)
gray2colora.ndim  # 3

# imshow : 행렬데이터를 이미지로 출력
cv2.imshow(title1,gray2gary)
cv2.imshow(title2,gray2color)
cv2.imshow(title3,gray2colora)

gray2gary = cv2.imread("images/read_color.jpg", cv2.IMREAD_GRAYSCALE)
gray2color = cv2.imread("images/read_color.jpg", cv2.IMREAD_COLOR)
gray2colora = cv2.imread("images/read_color.jpg", cv2.IMREAD_UNCHANGED)
if (gray2gary is None or gray2color is None) :
    raise Exception("영상파일 읽기 에러")
gray2gary.shape  # (300, 400)
gray2gary.ndim   # 2
gray2color.shape #(300, 400, 3)
gray2color.ndim  # 3
gray2colora.shape #(300, 400, 3)
gray2colora.ndim  # 3

cv2.imshow(title1,gray2gary)
cv2.imshow(title2,gray2color)
cv2.imshow(title3,gray2colora)

# 이미지 저장하기
image =  cv2.imread("images/read_color.jpg", cv2.IMREAD_COLOR)
# cv2.IMWRITE_JPEG_QUALITY(10) : 화질 선정
#                  0 ~ 100(95) : 숫자가 높으면 화질이 좋음
params_jpg = (cv2.IMWRITE_JPEG_QUALITY,10)   # 튜플,리스트 가능
# cv2.IMWRITE_PNG_COMPRESSION,9 : png 파일의 압축레벨 설정
#                      0 ~ 9(3) : 압축레벨이 높으면 이미지 용량이 작다.
params_png = [cv2.IMWRITE_PNG_COMPRESSION,9] # 튜플,리스트 가능
# imwrite : 배열데이터를 이미지파일로 저장
cv2.imwrite("images/write_test0.png",image) #204k 확장자에 맞춰서 이미지 저장
cv2.imwrite("images/write_test1.jpg",image) #51k
cv2.imwrite("images/write_test2.jpg",image,params_jpg) #6k
cv2.imwrite("images/write_test3.png",image,params_png) #171k
cv2.imwrite("images/write_test4.png",image) #352k
cv2.imwrite("images/write_test5.jpg",image,
            (cv2.IMWRITE_JPEG_QUALITY,100)) #90k









