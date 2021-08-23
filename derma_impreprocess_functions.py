import os
import cv2
import numpy as np

#--input: image folder path --output: image files list in formats: jpg, jpeg, png
def readDirectory(dir):
    os.chdir(dir)
    path = os.getcwd()
    imfilelist=[]
    for fn in os.listdir(path):
        if fn.endswith(('.jpg', '.jpeg', '.png')):
            imfilelist.append(fn)
    return imfilelist

# convert RGB to GRAY, then apply GaussianBlur, then reverse black & white
# i/p:original image directory path o/p:(original image, manipulated image)
def grayScale(origimdir):
    origim = cv2.imread(origimdir)
    imgtmp = cv2.cvtColor(origim, cv2.COLOR_BGR2GRAY)
    imgray = cv2.GaussianBlur(255-imgtmp, (5, 5), 0)
    return (origim,imgray)

# Calculates the centre of mass of a grayscale image
# i/p:grayscale image o/p:(Xcenter,Ycenter,Width,Height)
def centerOfMass(grayimage):
    imHeight, imWidth = grayimage.shape
    Sumx=np.int64(0)
    Sumy=np.int64(0)
    for j in range (imWidth):
        for i in range (imHeight):
            Sumx+=(grayimage[i,j]*(j+1))
            Sumy+=(grayimage[i,j]*(i+1))
    SumAllPixels=cv2.sumElems(grayimage)
    xCent = int(Sumx/SumAllPixels[0])
    yCent = int(Sumy/SumAllPixels[0])
    return (xCent,yCent,int(imWidth),int(imHeight))

# Crops the maximum possible square centralizing the center of mass
# i/p: (imgray,imorig,width,height,Xcenter,Ycenter) o/p:(Cropped gray, Cropped Original)
def cropSquare(imgray,imorig,w,h,cenX,cenY):
    a = min(h-cenY,cenY,w-cenX, cenX)
    cropgray = imgray[cenY-a:cenY+a, cenX-a:cenX+a]
    croporig = imorig[cenY - a:cenY + a, cenX - a:cenX + a]
    return cropgray,croporig

def savedirCreate(parent_dir,dir):
    path = os.path.join(parent_dir,dir)
    array = np.arange(100)
    savepath = 'blank'
    for i in array:
        if os.path.exists(path+str(i)):
            pass
        else:
            os.mkdir(path+str(i))
            if os.path.exists(path+str(i)):
                savepath = path+str(i)
                break
    return savepath

def imagesConvertSquare(imfilelist,imsavedirectory,save=True):
    savepath='blank'
    if save == True:
        savepath = savedirCreate(imsavedirectory,'Squared')
    for idx in imfilelist:
        print(idx)
        imorig, imgray = grayScale(idx)
        xcenter, ycenter, width, height = centerOfMass(imgray)
        imgraycircled = cv2.circle(imgray, (xcenter, ycenter), 10, (0, 0, 255), 2)
        cropgray, croporig = cropSquare(imgray, imorig, width, height, xcenter, ycenter)
        #cv2.imshow('grayscaled',cropgray)
        #cv2.waitKey(0)
        if save==True:
            impath = savepath+'\\'+ str(idx)
            print(impath)
            cv2.imwrite(impath, croporig)
    return 0
    #return (imgray, cropgray, croporig)