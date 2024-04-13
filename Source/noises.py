from random import random, gauss
import math
import cv2
import numpy as np
import os
def noises(type,M,N,a,b):
    if type=="uniform":
        n=np.zeros([M,N])
        for i in range(M):
            for j in range(N):
                n[i,j]=random()

        R=a+(b-a)*n
        return R
    elif type=="gaussian":
        n = np.zeros([M, N])
        for i in range(M):
            for j in range(N):
                n[i, j] = np.random.normal(0, 1)
        R = a + b * n
    elif type== "lognormal":
        n = np.zeros([M, N])
        for i in range(M):
            for j in range(N):
                n[i, j] = np.random.normal(0, 1)
        R=a*np.exp(b*n)
    elif type=="rayleigh":
        n = np.random.rand(M, N)
        R = a + (-b * np.log(1 - n)) ** 0.5
        return R
    elif type == "exponential":
        n = np.zeros([M, N])
        for i in range(M):
            for j in range(N):
                n[i, j] = random()
        k = -1 / a
        R = k * np.log(1 - n)
    elif type=="erlang":
        n = np.zeros([M, N])
        for i in range(M):
            for j in range(N):
                n[i, j] = random()
        k = -1 / a
        R = np.zeros([M, N])
        for i in range(b):
            R = R + k * np.log(1 - n)
    return R

def salt_papper(image,prob):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def add_periodic(img):
    orig = img
    sh = orig.shape[0], orig.shape[1]
    noise = np.zeros(sh, dtype='float32')

    X, Y = np.meshgrid(range(0, sh[0]), range(0, sh[1]))

    A = 40
    u0 = 45
    v0 = 50

    noise += A*np.sin(X*u0 + Y*v0)

    A = -18
    u0 = -45
    v0 = 50

    noise += A*np.sin(X*u0 + Y*v0)

    noiseada = orig+noise
    return(noiseada)

new_folder1 = 'mask_new/'
#new_folder2 = 'data_mask_rotated/'

# Check if the folders exist, if not, create them
if not os.path.exists(new_folder1):
    os.makedirs(new_folder1)
#if not os.path.exists(new_folder2):
    #os.makedirs(new_folder2)



image_files = [f for f in os.listdir('Dataset/mask/') if f.endswith(".png")]
for image_file in image_files:
        img_path = os.path.join('Dataset/mask/', image_file)
        img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        print(image_file)
        [M, N] = img.shape



    # Check if the folder exists, if not, create it




        noise1=img
        cv2.imwrite('sp_'+image_file, noise1)
        #gauss=noises('gaussian',M,N,0,50)
        noise2=img
        cv2.imwrite('g_'+image_file, noise2)
        #lognormal=60* noises('lognormal',M,N,1,0.25)
        noise3=img
        cv2.imwrite('ln_'+image_file, noise3)
        #uniform=noises('lognormal',M,N,0,100)
        noise4=img
        cv2.imwrite('u_'+image_file, noise4)

        #rayleigh=50*noises('rayleigh',M,N,0,1)
        noise5=img
        cv2.imwrite('r_'+image_file, noise5)

        #erlang=20*noises('erlang',M,N,2,5)
        noise6=img
        cv2.imwrite('er_'+image_file, noise6)

        #exponential=30*noises('exponential',M,N,1,1)
        noise7=img
        cv2.imwrite('ex_'+image_file, noise7)
'''

        rotated_img1 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        rotated_img2 = cv2.rotate(img, cv2.ROTATE_180)
        rotated_img3 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rotated_img4 = cv2.transpose(img)
    # Save the rotated image

        new_file_path = os.path.join(new_folder2,'90_'+image_file)
        cv2.imwrite(new_file_path, rotated_img1)
        new_file_path = os.path.join(new_folder2,'180_'+image_file)
        cv2.imwrite(new_file_path, rotated_img2)
        new_file_path = os.path.join(new_folder2,'270_'+image_file)
        cv2.imwrite(new_file_path, rotated_img3)
        new_file_path = os.path.join(new_folder2,'tr_'+image_file)
        cv2.imwrite(new_file_path, rotated_img4)



        #cv2.imshow('sp_'+image_file, noise)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
'''