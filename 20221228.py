# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 09:27:31 2022

@author: kimyh
"""

#이미지 파일을 읽어서 학습하기
import numpy as np, cv2
import matplotlib.pyplot as plt
size = (40, 40) #이미지 한 개의 크기
nclass, nsample = 10, 20
train_image = cv2.imread('images/train_numbers.png', cv2.IMREAD_GRAYSCALE)
train_image.shape
train_image = train_image[5:405, 6:806] #여백 제거
train_image.shape
'''
    cv2.threshold(train_image, 32, 255, cv2.THRESH_BINARY)
    cv2.threshold : 이미지의 값을 명확한 값으로 정리
    train_image : 400, 800 원본이미지
    32 : 임계값. 임계값 이상인 경우 설정값으로 변경
    255 : 설정값.
'''
cv2.threshold(train_image, 32, 255, cv2.THRESH_BINARY)
#np.hsplit : 수평분리. 열로 분리. 20개로 분리.
#np.vsplit : 수직분리. 행으로 분리. 10개로 분리.
cells = [np.hsplit(row, nsample) \
         for row in np.vsplit(train_image, nclass)]
print('cells 형태:', np.array(cells).shape)  #(10, 20, 40, 40)

def find_value_position(img, direct) :
    #cv2.reduce : 행 축소
    #direct : 0 : 열방향으로 연산하여 1행 축소
    #         1 : 행방향으로 1열 축소
    # ravel() : 1차원배열 변경
    project = cv2.reduce(img, direct, cv2.REDUCE_AVG).ravel()
    p0, p1 = -1, -1
    len = project.shape[0]
    for i in range(len) :
        if p0 < 0 and project[i] < 250 : p0 = i  #숫자시작좌표
        if p1 < 1 and project[len-i-1] < 250 : p1 = len-i-1  #종료좌표
    return p0, p1

# 숫자의 좌표값 리턴
def find_number(part) : #part : 숫자 한 개 이미지
    x0, x1 = find_value_position(part, 0)
    y0, y1 = find_value_position(part, 1)
    return part[y0:y1, x0:x1]  #숫자 이미지.

def place_middle(number, new_size) :
    h, w = number.shape[:2]
    big = max(h, w)
    #np.full(size,값,자료형) : 
    #   size의 배열에 값으로 채움. 값의 자료형은 np.float32
    square = np.full((big, big), 255, np.float32)
    #중간 좌표.
    dx, dy = np.subtract(big, (w, h)) // 2
    square[dy:dy + h, dx:dx + w] = number #가운데 숫자를 저장
    #숫자 정보의 데이터를 배열의 가운데에 저장
    #cv2.resize(square, new_size).flatten()
    #   square배열을 new_size 크기로 재 저장.
    #flatten() : 1차원 배열로 변경
    return cv2.resize(square, new_size).flatten()

# np.reshape(cells, (-1, 40, 40) :
#   cells 배열 : (10, 20, 40, 40) 형태
#   => 변경(-1, 40, 40) => (200, 40, 40)    
nums = [find_number(c) for c in np.reshape(cells, (-1, 40, 40))]
len(nums)
#학습데이터
trainData = np.array([place_middle(n, size) for n in nums])
labels = np.array([i for i in range(nclass) \
                   for j in range(nsample)], np.float32)
print('nums 형태:', np.array(nums).shape)
print('trainData 형태:', trainData.shape) #200* 1600 (40행*40열)
print('labels 형태:', labels.shape)
labels[:10]

#알고리즘
knn = cv2.ml.KNearest_create()
#학습하기
knn.train(trainData, cv2.ml.ROW_SAMPLE, labels)

plt.figure(figsize=(10,10))
for i in range(50) :
    #00~99 : 숫자당 5개씩 데이터를 예측하기
    test_img = cv2.imread\
        ('images/num/%d%d.png' % (i / 5, i % 5), cv2.IMREAD_GRAYSCALE)
    cv2.threshold(test_img, 128, 255, cv2.THRESH_BINARY)
    num = find_number(test_img) #숫자이미지
    data = place_middle(num, size) #숫자이미지 (40,40)크기에 가운데
    data = data.reshape(1, -1) #1,1600. 테스트데이터. 테스트이미지
    _, resp, _, _ = knn.findNearest(data, 5) #예측하기
    plt.subplot(10, 5, i+1)
    plt.axis('off')
    plt.imshow(num, cmap='gray')
    plt.title('resp ' + str(int(resp[0][0])))
plt.tight_layout()
plt.show()

# bitwise 함수 : 2개의 이미지 합하기
import numpy as np, cv2
image1 = np.zeros((300, 300), np.uint8) #300행, 300열 검은색 영상 생성
image2 = image1.copy()
image1.shape
cv2.imshow("image1", image1)
cv2.imshow("image2", image2)

h, w = image1.shape[:2] #300, 300
print(h,w)
cx, cy = w//2, h//2
print(cx, cy)
# 원을 표시
# (cx, cy) : 원의 중심 좌표
# 100 : 반지름
# 255 : 색상값
cv2.circle(image1, (cx,cy), 100, 255, -1)    #중심에 원 그리기
# 사각형 표시
# (0,0,cx,h) : (시작x좌표,시작y좌표,w,h)
# 255 : 색상값
cv2.rectangle(image2, (0,0,cx,h), 255, -1)
# 두 개의 이미지 합하기
image3 = cv2.bitwise_or(image1, image2)      #원소 간 논리합
image4 = cv2.bitwise_and(image1, image2)     #원소 간 논리곱
image5 = cv2.bitwise_xor(image1, image2)     #원소 간 배타적 논리합
image6 = cv2.bitwise_not(image1)             #행렬 반전
cv2.imshow("bitwise_or", image3);
cv2.imshow("bitwise_and", image4)
cv2.imshow("bitwise_xor", image5);
cv2.imshow("bitwise_not", image6)
cv2.waitKey(0)
