import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, img_as_ubyte
import time

start_time = time.time()

def histandpdf (img):
    # 返回直方图hist和概率密度函数PDF
    hist,bins=np.histogram(img.flatten(),256,[0,256])
    PDF=hist/len(img.flatten())
    return hist,bins,PDF

def mean_var_std_entropy(img,PDF):
    # 计算图像中位数，方差，标准差和熵  参数1为图像数组，
    data_mean=np.mean(img.flatten())
    data_var=np.var(img.flatten())
    data_std=np.std(img.flatten())
    # 计算熵时每个数都加上一个10的﹣10次方
    data_entropy=- sum([(p+10**-10) * np.log2(p+10**-10) for p in PDF])
    return data_mean,data_var,data_std,data_entropy

def cumulateandshow (img,hist,bins): 
    plt.figure()
    img_array=img.flatten()
    x=np.arange(0,256,1)
    plt.subplot(221)
    
    plt.bar(x,hist)
    
    # hist,bins = np.histogram(img_array,256,density=True)
    
    #计算累积分布函数
    cdf = hist.cumsum()
    
    #累计函数归一化
    cdf = 255*cdf/cdf[-1]
    
    #绘制累计分布函数

    #依次对每一个灰度图像素值（强度值）使用cdf进行线性插值，计算其新的强度值
    #interp（x，xp，yp） 输入原函数的一系列点（xp，yp），使用线性插值方法模拟函数并计算f（x）
    img2 = np.interp(img_array,bins[:256],cdf)
    hist2,bins2=np.histogram(img2,256,density=True)   
    #将压平的图像数组重新变成二维数组
    img2 = img2.reshape(img.shape).astype('uint8')

    # 显示均衡化之后的直方图图像
    plt.subplot(222)
    
    plt.bar(x,hist2)
    
    #显示原始图像
    
    plt.subplot(223)
    
    io.imshow(img)
    
    #显示变换后图像
    plt.subplot(224)
    
    io.imshow(img2)

    plt.show()

def function (img,m,n):
    # 量化函数 两个参数，img为传入图像的矩阵，
    # m为传入图像的bit，n为量化后的图像bit
    if m>=n:
        img=img>>(m-n) #将img内数据的二进制右移m-n位 ，进行m-n bit的量化
        img=img<<(m-n) #将上一步量化好的数据左移m-n位，再次映射回原图灰度级
        return img

def main():
    img1=io.imread('./img/lena.png',as_gray=True)
    img=img_as_ubyte(img1)

    hist,bins,PDF=histandpdf(img)
    x=np.arange(0,256,1)
    # 直方图
    plt.bar(x,hist)
    plt.show()
    #概率密度函数
    plt.plot(x,PDF)
    plt.show()
    # 计算中位数，方差，标准差和熵
    data_mean,data_var,data_std,data_entropy=mean_var_std_entropy(img,PDF)
    print('中位数是：%.2f'%data_mean)
    print('方差是：%.2f'%data_var)
    print('标准差是：%.2f'%data_std)
    print('熵是：%.2f'%data_entropy)
    #  直方图均衡化
    cumulateandshow(img,hist,bins)
    

    # 直方图量化
    # 1bit
    img_1bit=function(img,8,1)
    img_3bit=function(img,8,3)
    img_6bit=function(img,8,6)
    io.imsave('new1bit.png',img_1bit)
    io.imsave('new3bit.png',img_3bit)
    io.imsave('new6bit.png',img_6bit)




main()

print(time.time() - start_time)