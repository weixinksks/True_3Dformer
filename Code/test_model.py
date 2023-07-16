from myconfig import mypara
import numpy as np
from copy import deepcopy
import matplotlib as mpl
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from my_tools import cal_ninoskill2, runmean
from func_for_prediction import func_pre
import torch
import pandas as pd

import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import SST

mpl.use("TKAgg")
plt.rc("font", family="Arial")
mpl.rc("image", cmap="RdYlBu_r")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"




# 使用以下的代码来获取训练得到的模型
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == ".pkl":
                L.append(os.path.join(root, file))
    return L




# --------------------------------------------------------
# 定义一些全局变量，模型的路径，需要去更改模型。
files = ("D:/AIProject/Code_of_3D-Geoformer/Data/tmp_module/Geoformer.pkl")
print(files)
# file_num = len(files)
# print(file_num)

# 设置输出数据的长度
lead_max = mypara.output_length



# 需要重新进行预测结果的可视化，包括时间纬度和空间纬度上的可视化，
# 需要在时间纬度上进行可视化制作，即在单点数据上或者是在场数据上,利用时间纬度进行可视化，
# 需要在空间纬度上进行可视化制作，即需要对于数据在空间维度上进行可视化，




# 首先第一步定义相关的路径，将测试数据输入到模型当中。
adr_data  = ("E:/SST_DATA/pacific/")



# 是否可以在func_pre中直接返回关于太平洋模式预报数据的经纬度信息和掩码信息
adr_data_err = "E:/SST_DATA/pacific/err/20190101/sst.000.nc"

file_object_xr = xr.open_dataset(adr_data_err)

print(file_object_xr)

# # 需要获取相关的数据，从而得到对应的经纬度信息和掩码信息
GLOBAL_LAT = file_object_xr.variables['lat'].values
GLOBAL_LON = file_object_xr.variables['lon'].values
mask_ocean_data = file_object_xr["sst"][0,:].isnull().values





# 测试文件进行更改，使得数据符合测试功能所定义的数据类型

(cut_var_pred, cut_var_true)= func_pre(
    mypara=mypara,
    adr_model=files,
    adr_data=adr_data
)

print(cut_var_pred)
print(cut_var_true)
# 打印处理的数据形状是[10,25,1,660,780]
print(cut_var_pred.shape)
print(cut_var_pred.shape[0]) # 这里输出的时相关预测数据，[ngroup,input_length,time,lat,lon]






for i in range(lead_max):

    # 首先计算预测图像的一个平均绝对误差

    y_pred = cut_var_pred[0,i,0,:,:]
    # y_true = cut_var_true[0,i,0,:,:]
    err_abs = abs(y_pred)
    mae = np.nanmean(err_abs)
    print('pre_Mae:', mae)


    # 之后计算真实值的平均绝对误差
    # y_pred = cut_var_pred[0,i,0,:,:]
    y_true = cut_var_true[0,i,0,:,:]
    err_abs = abs(y_true)
    mae = np.nanmean(err_abs)
    print('true_Mae:', mae)


    # 需要循环计算相关的预测指标，即平均绝对误差
    y_pred = cut_var_pred[0,i,0,:,:]
    y_true = cut_var_true[0,i,0,:,:]
    err_abs = abs(y_pred-y_true)
    mae = np.nanmean(err_abs)
    print('err_Mae:', mae)

    # 这里计算RMSE损失（均方根误差）
    # 采用torch本身封装好的损失函数接口进行计算
    y_pred = torch.tensor(cut_var_pred[0,i,0,:,:])
    y_true = torch.tensor(cut_var_true[0,i,0,:,:])
    criterion = torch.nn.MSELoss()
    mse_loss = criterion(y_pred, y_true)
    rmse_loss = torch.sqrt(mse_loss)
    print('RMSE_Loss:', rmse_loss.item())


    # 进行图像绘制
    vmin = -3.0
    vmax = 3.0
    interval = 0.5

    # 创建自定义的colorbar
    colortable_name = "my_colortable"   #定义颜色表名称
    cmap_data = np.loadtxt('D:/AIProject/Code_of_3D-Geoformer/Drawing/hotcolr_19lev.rgb') / 255.0
    cmap = colors.ListedColormap(cmap_data,colortable_name)

    # 构造两个子图，一个显示预测图像，一个显示真实图像
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))






    # 全部数据，包括陆地和海洋数据
    pre_data =cut_var_pred[0,i,0,:,:] 
    true_data =cut_var_true[0,i,0,:,:]
    # data =file_object_xr["sst"][0,:,:]
    # data =cut_var_pred[0,i,0,:,:]-cut_var_true[0,i,0,:,:]
    # # 需要利用掩码信息将陆地信息和海洋信息提取出来
    land_data = np.ma.masked_array(pre_data, ~mask_ocean_data)
    
    axes[0].imshow(np.flipud(land_data), cmap='gray', vmin=vmin, vmax=vmax, alpha=0.5)

    # 绘制数据图像，并选择自定义的colorbar
    ocean_data = np.ma.masked_array(pre_data, mask_ocean_data)
    im1 = axes[0].imshow(np.flipud(ocean_data), cmap=cmap, vmin=vmin, vmax=vmax) 

    # # 需要利用掩码信息将陆地信息和海洋信息提取出来
    land_data = np.ma.masked_array(true_data, ~mask_ocean_data)
    
    axes[1].imshow(np.flipud(land_data), cmap='gray', vmin=vmin, vmax=vmax, alpha=0.5)

    # 绘制数据图像，并选择自定义的colorbar
    ocean_data = np.ma.masked_array(true_data, mask_ocean_data)
    im2 = axes[1].imshow(np.flipud(ocean_data), cmap=cmap, vmin=vmin, vmax=vmax) 



    # 设置刻度位置

    lon_ticks = np.arange(file_object_xr['lon'][0], file_object_xr['lon'][-1]+1, 5)
    lat_ticks = np.arange(file_object_xr['lat'][0], file_object_xr['lat'][-1]+1, 5)

    axes[0].set_xticks(np.linspace(0, pre_data.shape[1], len(lon_ticks)), lon_ticks)
    axes[0].set_yticks(np.linspace(0, pre_data.shape[0], len(lat_ticks)), np.flipud(lat_ticks))

    axes[1].set_xticks(np.linspace(0, pre_data.shape[1], len(lon_ticks)), lon_ticks)
    axes[1].set_yticks(np.linspace(0, pre_data.shape[0], len(lat_ticks)), np.flipud(lat_ticks))


    # 设置X轴和Y轴刻度标签
    x_tick_labels = ['98°E', '105°E','110°E', '115°E', '120°E', '125°E','130°E','135°E','140°E','145°E', '150°E', '155°E', '160°E','165°E','170°E','175°E']
    y_tick_labels = ['48°N','40°N','35°N', '30°N', '25°N', '20°N','15°N','10°N','5°N','0°N','-5°N','-10°N','-15°N','-20°N']

    axes[0].set_xticklabels(x_tick_labels,fontsize = 12,rotation = 45)
    axes[0].set_yticklabels(y_tick_labels)

    axes[1].set_xticklabels(x_tick_labels,fontsize = 12,rotation = 45)
    axes[1].set_yticklabels(y_tick_labels)


    # # 显示颜色条
    fig.colorbar(im1,ax=axes[0], ticks=np.arange(vmin, vmax + interval, interval))
    fig.colorbar(im2,ax=axes[1], ticks=np.arange(vmin, vmax + interval, interval))
    # # 在图形上方添加共享的x轴标签
    fig.text(0.5, 0.05, 'Lon', ha='center')
    
    # 在图形左侧添加共享的y轴标签
    fig.text(0.02, 0.5, 'Lat', va='center', rotation='vertical')

    # 显示图形
    plt.show()
print("*************" * 8)