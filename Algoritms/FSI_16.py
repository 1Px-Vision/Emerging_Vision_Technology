import numpy as np
import math
import cv2
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import scipy.io
import time
from time import time
from PIL import Image as im

#from RefinedTramsmission import Refinedtransmission
#from getAtomsphericLight import getAtomsphericLight
#from getGbDarkChannel import getDarkChannel
#from getTM import getTransmission
#from sceneRadiance import sceneRadianceRGB

def FSPIReconstruction( I, nStepPS, Phaseshift ):
    if ((nStepPS == 4) & (Phaseshift == 90)):
        a1=I[:,:,0]-I[:,:,1]
        b1=I[:,:,2]-I[:,:,3]
        spec = (a1+1j*b1).T


    spec = completeSpec(spec)
    img  = np.real(np.fft.ifft2(np.fft.ifftshift(spec)))

    return img, spec

def completeSpec(halfSpec):
   if (len(halfSpec.shape)==2):
       mRow, nCol = halfSpec.shape[0],halfSpec.shape[1]
   if (len(halfSpec.shape)==1):
       mRow=1
       nCol=halfSpec.shape[0]

   fullSpec =np.zeros((halfSpec.shape),dtype=np.complex_)

   if (((mRow%2) == 1) & ((nCol%2) == 1)): 
      halfSpec = halfSpec + np.rot90(np.conj(halfSpec), 2)
      if ((mRow/2)<1):
          halfSpec[0,(math.ceil(nCol/2)-1)] = (halfSpec[0,(math.ceil(nCol/2)-1)])/2
          
      if ((nCol/2)<1):      
          halfSpec[(math.ceil(mRow/2)-1),0] = (halfSpec[(math.ceil(mRow/2)-1),0])/2
          
      if (((mRow/2)>1) & ((nCol/2)>1)):   
          halfSpec[(math.ceil(mRow/2)-1),(math.ceil(nCol/2)-1)] = (halfSpec[(math.ceil(mRow/2)-1),(math.ceil(nCol/2)-1)])/2
          
      fullSpec = halfSpec

   else:
    if (((mRow%2) == 0) & ((nCol%2) == 0)):                   # Even * Even
        RightBottomHalfSpec = halfSpec[1:,1:]
        RightBottomFullSpec = completeSpec(RightBottomHalfSpec)
        
        fullSpec[1:,1:] = RightBottomFullSpec
        
        TopLine = np.array([halfSpec[0, 1:]])
        TopLine = completeSpec(TopLine)
        fullSpec[0, 1:] = TopLine
        
        LeftColumn = np.array([halfSpec[1:, 0]]).T
        LeftColumn = completeSpec(LeftColumn)
        fullSpec[1:, 0]= LeftColumn[:,0]
        
        fullSpec[0,0] = halfSpec[0,0]

    else:
        if (((mRow% 2) == 1) & ((nCol% 2) == 0)):   # ODD * EVEN
            LeftColumn = halfSpec[:, 0]
            LeftColumn = completeSpec(LeftColumn)
            
            RightHalfSpec = np.array([halfSpec[:,1:]])
            RightFullSpec = completeSpec(RightHalfSpec)
            
            fullSpec[:, 0] = LeftColumn
            fullSpec[:, 1:] = RightFullSpec
        else:                                         # EVEN * ODD
            halfSpec = halfSpec.T
            fullSpec = completeSpec(halfSpec)
            fullSpec = fullSpec.T


   return fullSpec

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

if __name__=="__main__":
    m_1 = scipy.io.loadmat('parameter_fsi.mat')
    m_2=scipy.io.loadmat('dat_FSI_16.mat')#fsi_dat.mat profile_dat.mat

   
    img_dat=np.zeros((128))
   


    nCoeft=m_1['nCoeft'][0][0]
    iRow=m_1['iRow1'][0]
    jCol=m_1['jCol1'][0]
    cont=0
    cont1=0
    nStepPS=4
    Phaseshift=90
    
    intensity_mat=np.zeros((16,16,nStepPS))
    cont1=0
    for ii in range(32):
        for jj in range(nStepPS):
            intensity_mat[iRow[ii]-1,jCol[ii]-1,jj]=img_dat[cont1]
            cont1=cont1+1
            

        
    IntensityMat=np.zeros((16,16,4))
    IntensityMat[:,:,0]=np.reshape(m_2['measurement'][:,:,0],(16,16))#np.reshape(intensity_mat[:,:,0],(8,8))##temp11
    IntensityMat[:,:,1]=np.reshape(m_2['measurement'][:,:,1],(16,16))#np.reshape(intensity_mat[:,:,1],(8,8))#
    IntensityMat[:,:,2]=np.reshape(m_2['measurement'][:,:,2],(16,16))#np.reshape(intensity_mat[:,:,2],(8,8))#
    IntensityMat[:,:,3]=np.reshape(m_2['measurement'][:,:,3],(16,16))#np.reshape(intensity_mat[:,:,3],(8,8))#
    
    start = time()
    img, spec=FSPIReconstruction( IntensityMat, nStepPS, Phaseshift )#intensity_mat  IntensityMat
    img_spi_b=img
    
    img1=filter_bilateral(img, 0.8,0.2)#0.48
    img2=((img1 - img1.min()) * (1/(img1.max() - img1.min()) * 255)).astype('uint8')
    img2=cv2.resize(img2,(8,8),interpolation=cv2.INTER_CUBIC)
    img2=object_fscnn(img2)
    end = time()
    time_pro=end-start
    print('Time FSI_CNN={} ms'.format(time_pro*1000))


    img_spi_b=np.rot90(img_spi_b)
    img_spi_b=np.rot90(img_spi_b)
    img_spi_b=np.rot90(img_spi_b)

    img2=np.rot90(img2)
    img2=np.rot90(img2)
    img2=np.rot90(img2)
    
    plt.figure(figsize=(32,32))
    plt.subplot(121)
    plt.imshow(img_spi_b,cmap='gray');
    plt.title("GPU-OMP")

    plt.subplot(122)
    plt.imshow(img2,cmap='gray');
    plt.title("FSRCNN");
    plt.show()      

  
