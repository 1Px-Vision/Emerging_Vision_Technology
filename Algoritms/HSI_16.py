import numpy as np
import math
import cv2
from numpy import r_
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import time
from time import time
import scipy.io
from PIL import Image





def filter_bilateral( img_in, sigma_s, sigma_v, reg_constant=1e-8 ):
   
    # check the input
    #if not isinstance( img_in, np.ndarray ) or img_in.dtype != 'float32' or img_in.ndim != 2:
     #   raise ValueError('Expected a 2D numpy.ndarray with float32 elements')

    # make a simple Gaussian function taking the squared radius
    gaussian = lambda r2, sigma: (np.exp( -0.5*r2/sigma**2 )*3).astype(int)*1.0/3.0

    win_width = int( 3*sigma_s+1 )

    wgt_sum = np.ones( img_in.shape )*reg_constant
    result  = img_in*reg_constant

    for shft_x in range(-win_width,win_width+1):
        for shft_y in range(-win_width,win_width+1):
            # compute the spatial weight
            w = gaussian( shft_x**2+shft_y**2, sigma_s )

            # shift by the offsets
            off = np.roll(img_in, [shft_y, shft_x], axis=[0,1] )

            # compute the value weight
            tw = w*gaussian( (off-img_in)**2, sigma_v )

            # accumulate the results
            result += off*tw
            wgt_sum += tw

    # normalize the result and return
    return result/wgt_sum

def object_fscnn(image):   
    model='FSRCNN_x4.pb'
    modelName='fsrcnn'
    modelScale=4
    
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model)
    sr.setModel(modelName, modelScale)
    
    upscaled = sr.upsample(image)
    return upscaled

def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max

def getHSPIReconstruction( dataMat, nStep ):
    if (nStep == 2):
        spec = dataMat[:,:,0] - dataMat[:,:,1]
        img  = fwht2d(spec)
    
    return img, spec


def fwht2d(xx):
   N = len(xx)
   xx1=np.zeros((N,N))
   for i in range(N):
        xx1[i,:] = fhtseq_inv(xx[i,:]) 
    

   xx =np.zeros((N,N))
   for j in range(N):
     xx[:,j] = fhtseq_inv(xx1[:,j].T).T

   
   return xx

def fhtseq_inv(data):
    N = len(data)
    L = np.log2(N)
    if ((L-np.floor(L)) > 0.0):
        raise (ValueError, "Length must be power of 2")
    x=bitrevorder(data)

    k1=N
    k2=1
    k3=int(N/2)
    for i1 in range(1,int(L+1)):  #Iteration stage
        L1=1
        for i2 in range(1,int(k2+1)):
            for i3 in range(1,int(k3+1)):
                ii=int(i3+L1-1) 
                jj=int(ii+k3)
                temp1= x[ii-1]
                temp2 = x[jj-1]
                if (i2 % 2) == 0:
                    x[ii-1] = temp1 - temp2
                    x[jj-1] = temp1 + temp2
                    
                else:
                    x[ii-1] = temp1 + temp2
                    x[jj-1] = temp1 - temp2
                    
            L1=L1+k1
        k1 = round(k1/2)
        k2 = k2*2
        k3 = round(k3/2)

    return (1/N)*x

def bitrevorder(x):
    temp_x=np.arange(0,len(x))
    temp_y=digitrevorder(temp_x,2)
    return x[temp_y]


def digitrevorder(x,base):
    x = np.asarray(x)
    rem = N = len(x)
    L = 0
    while 1:
        if rem < base:
            break
        intd = rem // base
        if base*intd != rem:
            raise (ValueError, "Length of data must be power of base.")
        rem = intd
        L += 1
    vec = r_[[base**n for n in range(L)]]
    newx = x[np.newaxis,:]*vec[:,np.newaxis]
    # compute digits
    for k in range(L-1,-1,-1):
        newx[k] = x // vec[k]
        x = x - newx[k]*vec[k]
    # reverse digits
    newx = newx[::-1,:]
    x = 0*x
    # construct new result from reversed digts
    for k in range(L):
        x += newx[k]*vec[k]
    return x

if __name__=="__main__":
    m_parameter=scipy.io.loadmat('parameter_hsi.mat')
    m_image=scipy.io.loadmat('dat_HSI_16.mat')
   

    nStep=2
    nCoeft=256
    iRow=m_parameter['iRow1'][0]
    jCol=m_parameter['jCol1'][0]
                                

    intensity_mat=np.zeros((16,16,nStep))
    IntensityMat=np.zeros((16,16,nStep))
    
    cont1=0
    for ii in range(nCoeft):
        for jj in range(nStep):
            intensity_mat[iRow[ii]-1,jCol[ii]-1,jj]=m_image['measurement'][0,cont1]#img_dat[cont1]/(32)
            cont1=cont1+1
    

    start = time()
    img_spi, spec=getHSPIReconstruction(intensity_mat, nStep )#IntensityMat  intensity_mat
    img_spi=((img_spi - img_spi.min()) * (1/(img_spi.max() - img_spi.min()) * 255)).astype('uint8')
    img_spi_b=img_spi
    img_spi=im2double(img_spi)
    
    img1=filter_bilateral(img_spi,0.8,0.3)
    
    img2=((img1 - img1.min()) * (1/(img1.max() - img1.min()) * 255)).astype('uint8')
    img2[np.where(img2>100)]=255
    img2=object_fscnn(img2)
    end = time()
    time_pro=end-start
    print('Time WH1={} ms'.format(time_pro*1000))


    plt.figure(figsize=(32,32))
    plt.subplot(121)
    plt.imshow(img_spi_b,cmap='gray');
    plt.title("GPU-OMP")

    plt.subplot(122)
    plt.imshow(img2,cmap='gray');
    plt.title("FSRCNN");
    plt.show()                   
               
