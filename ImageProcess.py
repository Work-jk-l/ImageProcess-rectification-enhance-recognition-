import os
import cv2
import math
import numpy as np

out_dir="./report.txt"
imageFileNames='./SourceImg'

def detectFindContours(cannyImg,dilate=False):
    image,contours,hierarchy=cv2.findContours(cannyImg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    areas=[]
    contoursLength=[]
    outCurve=[]
    curves=[]
    areaNum=[]
    #cv2.imshow("image",cannyImg)
    #cv2.waitKey(0)
    for i in range(len(contours)):
        areas.append(cv2.contourArea(contours[i]))

    for i in range(len(areas)-1,-1,-1):
        length=cv2.arcLength(contours[i],closed=True)
        if(length<6000):
            continue
        contoursLength.append(length)
        outCurve=cv2.approxPolyDP(contours[i],0.02*length,True)
        curveArea=abs(cv2.contourArea(outCurve,True))
        print("curveArea %d",curveArea)
        if(dilate==False):
            if(len(outCurve) % 4 ==0 and curveArea > 5000):
                curves.append(outCurve)
                areaNum.append(curveArea)
        else:
            if(len(outCurve) >=4 and curveArea>= 10000):
                curves.append(outCurve)
                areaNum.append(curveArea)
    contoursLength.sort()
    return outCurve,curves,contoursLength,areaNum,contours

def findDisMinus(curves):
    pointsMaxX = 0.0
    pointsMaxY = 0.0
    for m in range(len(curves)):
        outCurve=curves[m]
        for i in range(len(outCurve)-1):
            for j in range(i+1,len(outCurve)):
                x=abs(outCurve[i][0][0]-outCurve[j][0][0])
                y=abs(outCurve[i][0][1]-outCurve[j][0][1])
                if(x>pointsMaxX):
                    pointsMaxX=int(x)
                if(y>pointsMaxY):
                    pointsMaxY=int(y)
    return pointsMaxX,pointsMaxY

def detectPoints(curves,outCurve,image):
    points_=[]
    distancesToZero=[]
    distancesToRightDown=[]
    distancesToRightUp=[]
    distancesToLeftDown=[]
    if(outCurve):
        outCurve=None

    for j in range(len(curves)):
        outCurve=curves[j]
        for i in range(len(outCurve)):
            distToZero=pow(outCurve[i][0][0],2)+pow(outCurve[i][0][1],2)
            distToRightDown = pow(outCurve[i][0][0] - image.shape[1], 2) + pow(outCurve[i][0][1] - image.shape[0], 2)
            distToRightUp = pow(outCurve[i][0][0] - image.shape[1], 2) + pow(outCurve[i][0][1] - 0, 2)
            distToLeftDown = pow(outCurve[i][0][0] - 0, 2) + pow(outCurve[i][0][1] - image.shape[0], 2)
            distancesToZero.append(distToZero)
            distancesToRightDown.append(distToRightDown)
            distancesToRightUp.append(distToRightUp)
            distancesToLeftDown.append(distToLeftDown)

    if(len(outCurve)>0):
        outCurve=None
    distancesToZero.sort()
    distancesToRightDown.sort()
    distancesToRightUp.sort()
    distancesToLeftDown.sort()

    pointToRightDownMax=[]
    pointToZeroMax=[]
    pointToRightUpMax=[]
    pointToLeftDownMax=[]

    for j in range(len(curves)):
        outCurve=curves[j]
        for i in range(len(outCurve)):
            distToZero = pow(outCurve[i][0][0], 2) + pow(outCurve[i][0][1], 2)
            distToRightDown = pow(outCurve[i][0][0] - image.shape[1], 2) + pow(outCurve[i][0][1] - image.shape[0], 2)
            distToRightUp = pow(outCurve[i][0][0] - image.shape[1], 2) + pow(outCurve[i][0][1] - 0, 2)
            distToLeftDown = pow(outCurve[i][0][0] - 0, 2) + pow(outCurve[i][0][1] - image.shape[0], 2)

            if (distToZero == distancesToZero[len(distancesToZero) - 1]):
                pointToZeroMax = outCurve[i]

            if (distToRightDown == distancesToRightDown[len(distancesToRightDown) - 1]):
                pointToRightDownMax = outCurve[i]

            if (distToRightUp == distancesToRightUp[len(distancesToRightUp) - 1]):
                pointToRightUpMax = outCurve[i]

            if (distToLeftDown == distancesToLeftDown[len(distancesToLeftDown) - 1]):
                pointToLeftDownMax = outCurve[i]

    points_ = np.float32([pointToRightDownMax,pointToLeftDownMax,pointToRightUpMax,pointToZeroMax])
    points_.reshape((4,2))
    return points_

def rotateImage(pointsMaxX,pointsMaxY,points_,img,imgName):
    points=np.float32([[0,0],[pointsMaxX,0],[0,pointsMaxY],[pointsMaxX,pointsMaxY]])

    M = cv2.getPerspectiveTransform(points_, points)
    rotateImg = cv2.warpPerspective(img, M, (pointsMaxX, pointsMaxY))

    writeName = "./ReduceParames/"
    if (not os.path.exists(writeName)):
        os.mkdir(writeName)
    writeName = os.path.join(writeName, imgName)

    cv2.imwrite(writeName, rotateImg)


def Method():
    outTxtFileName="./report.txt"
    fp=open(outTxtFileName,"w+")

    imgNames=os.listdir(imageFileNames)
    for imgName in imgNames:
        imgPath=os.path.join(imageFileNames,imgName)
        img=cv2.imread(imgPath)
        imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        medianImg=cv2.medianBlur(imgGray,33)
        cannyImg=cv2.Canny(medianImg,30,80)
        outCurve,curves,contoursLength,areaNum,contours=detectFindContours(cannyImg,False)

        if(len(curves)>=1):
            areaNum.sort()
            curves1=[]
            for m in range(len(curves)):
                outcurve1=curves[m]
                curveArea=abs(cv2.contourArea(outcurve1,True))
                if(curveArea==areaNum[len(areaNum)-1]):
                   curves1.append(curves[m])

            if(len(outCurve)>0):
                outCurve=None
            pointsMaxX=0.0
            pointsMaxY=0.0

            pointsMaxX,pointsMaxY=findDisMinus(curves1)

            points_=detectPoints(curves1,outCurve,img)

            rotateImage(pointsMaxX,pointsMaxY,points_,img,imgName)
            fp.write("Method:面积最大的点围成的多边形:\n" + imgName)
        else:
            elements=cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
            dilateImage=cv2.dilate(cannyImg,elements,(-1,-1))
            # cv2.imshow("dilate",dilateImage)
            # cv2.waitKey(0)
            outCurve, curves, contoursLength, areaNum, contours=detectFindContours(dilateImage,True)
            if(len(curves)>=1):
                areaNum.sort()
                curves1 = []
                for m in range(len(curves)):
                    outCurve1=curves[m]
                    contourArea=abs(cv2.contourArea(outCurve1,True))
                    if(contourArea==areaNum[len(areaNum)-1]):
                        curves1.append(curves[m])

                if(len(outCurve)>0):
                    outCurve=None
                pointsMaxX,pointsMaxY=findDisMinus(curves1)
                points_ = detectPoints(curves1, outCurve, img)

                rotateImage(pointsMaxX, pointsMaxY, points_, img, imgName)
                fp.write("Method:膨胀后面积最大的点围成的多边形:\n" + imgName)
            else:#找到周长最大的多边形
                for i in range(len(contours)):
                    length=cv2.arcLength(contours[i],True)
                    outCurve=cv2.approxPolyDP(contours[i],0.02*length,True)

                    if(length==contoursLength[len(contoursLength)-1]):
                        curves.append(outCurve)
                        break

                pointsMaxX, pointsMaxY = findDisMinus(curves)
                points_ = detectPoints(curves, outCurve, img)

                rotateImage(pointsMaxX, pointsMaxY, points_, img, imgName)
                fp.write("Method:膨胀后周长最大的点围成的多边形:\n" + imgName)
    fp.close()

def sauvola(img,imgGary,k,kernel_width,imageName,imgGaryPath):
    tempImage=imgGary.copy()
    # sumIntegral=np.zeros((img.shape[0],img.shape[1]),dtype=np.int32)
    # sumSquare=np.zeros((img.shape[1],img.shape[0]),dtype=np.int32)

    intergralImage=cv2.integral(imgGary)
    sum,squareImage=cv2.integral2(imgGary)
    #print(type(squareImage))
    #print(type(intergralImage))


    for i in range(imgGary.shape[0]):       #shape[0]:height
        for j in range(imgGary.shape[1]):   #shape[1]:width
            xmin=int(max(0,j-kernel_width))
            ymin=int(max(0,i-kernel_width))
            xmax=int(min(imgGary.shape[1]-1,j+kernel_width))
            ymax=int(min(imgGary.shape[0]-1,i+kernel_width))
            area=(xmax-xmin+1)*(ymax-ymin+1)
            if(area<0):
                print("Error in the area: %d",area)
                return
            if(xmin==0 and ymin==0): #the first pixel
                intergralPxielValue=intergralImage[ymax,xmax]
                squarePixelValue=squareImage[ymax,xmax]
            elif(xmin==0 and ymin>0): #the first col
                intergralPxielValue=intergralImage[ymax,xmax]-intergralImage[ymin-1,xmax]
                squarePixelValue=squareImage[ymax,xmax]-squareImage[ymin-1,xmax]
            elif(xmin>0 and ymin==0): #the first row
                intergralPxielValue=intergralImage[ymax,xmax]-intergralImage[ymax,xmin-1]
                squarePixelValue=squareImage[ymax,xmax]-squareImage[ymax,xmin-1]
            else: #the rest pixel
                mainDiagonalIntergralPixelValue=intergralImage[ymax,xmax]+intergralImage[ymin-1,xmin-1]
                counterDiagonalIntergralPixelValue=intergralImage[ymin-1,xmax]+intergralImage[ymax,xmin-1]
                intergralPxielValue=mainDiagonalIntergralPixelValue-counterDiagonalIntergralPixelValue

                mainDiagonalSquarePixelValue = squareImage[ymax, xmax] + squareImage[ymin - 1, xmin - 1]
                counterDiagonalSquarePixelValue = squareImage[ymin - 1, xmax] + squareImage[ymax, xmin - 1]
                squarePixelValue = mainDiagonalSquarePixelValue - counterDiagonalSquarePixelValue

            #注意防止越界
            mean=intergralPxielValue/area
            a=np.longlong(0)
            a=math.pow(intergralPxielValue,2)
            a=a/area
            stdDev=math.sqrt((squarePixelValue-a)/(area-1))
            std = math.sqrt((squarePixelValue - math.sqrt(intergralPxielValue) / area) / (area - 1))
            threshold=mean*(1+k*((stdDev/128)-1))
            if(imgGary[i,j]>threshold):
                tempImage[i,j]=255
                for m in range(3):
                    img[i,j,m]=255
            if(imgGary[i,j]<threshold):
                tempImage[i,j]=0

    writeGrayImgName="./imageEnhance"
    writeImgName="./imageEnhance"
    if(not os.path.exists(writeGrayImgName)):
        os.mkdir(writeGrayImgName)
    if(not os.path.exists(writeImgName)):
        os.mkdir(writeImgName)
    writeGrayImgName=os.path.join(writeGrayImgName,imgGaryPath)
    writeImgName=os.path.join(writeImgName,imageName)
    cv2.imwrite(writeImgName,img)
    cv2.imwrite(writeGrayImgName,tempImage)

def ImageEnhance():
    fileName="./ReduceParames/"
    imageNames=os.listdir(fileName)
    for imageName in imageNames:
        imgPath=os.path.join(fileName,imageName)
        #imgGaryPath=os.path.join("gary_",imageName)
        imgGaryPath="gray_"+imageName
        img=cv2.imread(imgPath)
        imgGary=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        k=0.05
        kernel_width=500
        x=kernel_width//2
        sauvola(img,imgGary,k,x,imageName,imgGaryPath)

if __name__ == '__main__':
   # Method()  #找顶点
    ImageEnhance()
