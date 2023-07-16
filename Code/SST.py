
import xarray as xr
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from torch.utils.data import Dataset, DataLoader
import torch
from pytorch_lightning import LightningDataModule
import os
from omegaconf import OmegaConf
import numpy as np
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset
import xarray as xr
import random
import pandas as pd
import netCDF4 as nc
import itertools
from sklearn.preprocessing import MinMaxScaler

""" 对于给出的数据(多个nc文件),需要将这些文件循环合并为一个nc文件,对于具体的训练集，验证集相关的数据集形状如下：
    训练集:原模型采用的是CMIP6的数据格式,具体的形状为
    (不同的数据采集模式个数,采集之后的月平均数month,通道数[这里是7层海洋温度数据和径向,纬向风应力数据]channel,经度数据lat,纬度数据lon}),
    因此需要将训练集指定为('pattern', 'date', 'time', 'lat', 'lon') = (1,281,4,30,26)[这里的样本数据是2009年的数据]。
    
    对于验证集，其中必须包含输入历史数据，输出预测数据。
    两者数据形状按照原来的模型指定为data_in = (样本数[ngroup],历史时间长度[input_length],lev,lat,lon),data_out=(样本数[ngroup],历史时间长度[output_length],lev,lat,lon)
    基于此，需要将验证集设置为相同的格式。   """



# # # # 制作数据集，包括训练集，验证集，测试集


# # # 获取数据地址
# _CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))


# cfg = OmegaConf.create()
# cfg.root_dir = os.path.abspath(os.path.realpath(os.path.join(_CURR_DIR, "..")))
# cfg.datasets_dir =os.path.join(cfg.root_dir, "data")




# # 得到默认的数据地址
# default_enso_dir =cfg.datasets_dir
# print(default_enso_dir)

# # 读取得到相关的lat，lon信息
# default_tmp_dir = os.path.join(default_enso_dir, "2009\\20090326.nc")
# # print(default_tmp_dir)
# tmp_data = xr.open_dataset(default_tmp_dir)
# print(tmp_data)
# GLOBAL_LAT = tmp_data.variables['lat'].values
# GLOBAL_LON = tmp_data.variables['lon'].values
# err_data = tmp_data['err']
# mask_ocean_data = err_data[0,:].isnull().values


# # 制作一个训练集
# # 循环得到每个nc文件的路径
# # 定义一个含有多个路径的列表


# years = ["2012","2013","2014"]
# # # # 在循环的过程中制作相关的训练输入数据和输出数据
# # # # 在这里默认将数据全部置为nan，除非得到第一个数据
# # 首先记录一个样本数据2009\\20090326.nc  这个数据

# # err_first_tmp_data = tmp_data.variables['err'].values
# # err_time_data =  err_first_tmp_data[0,:]
# # # 进行0值填充
# # err_filled_data = np.nan_to_num(err_time_data,nan = 0)
# # print(err_filled_data)
# # print(err_filled_data.shape)
# # # 需要对与数据进行reshape，将[625,721] -> [1,样本数,lev(1)，lat(625),lon(721)]
# # err_data = err_filled_data.reshape(1, 1, 1, err_filled_data.shape[0], err_filled_data.shape[1])
# # print(err_data.shape)
# # ds_all = xr.Dataset({'err':(['model','date','time','lat','lon'],err_data),'mask':(['lat','lon'],mask_ocean_data)},coords={'time':[1],'lat':GLOBAL_LAT,'lon':GLOBAL_LON})

# ds_all = xr.Dataset()

# for year in years:
#     default_err_dir = os.path.join(default_enso_dir, year)
#     # print(default_enso_dir)
#     paths = [os.path.join(default_err_dir, f) for f in os.listdir(default_err_dir) if f.endswith('.nc')]
#         # print(paths)  
#     path_number = len(paths)
#     print(paths)
#     print(path_number)
#     for path in paths:
#         ds = xr.open_dataset(path)
#         # 得到未处理nan值的数据
#         err_no_fillnan_data = ds.variables['err'].values
#         # 选择time = 0 的数据
#         err_time_data =  err_no_fillnan_data[0,:]
#         # 进行0值填充
#         err_filled_data = np.nan_to_num(err_time_data,nan = 0)
#         print(err_filled_data)
#         print(err_filled_data.shape)
#         # 需要对与数据进行reshape，将[625,721] -> [1,样本数,lev(1)，lat(625),lon(721)]
#         err_data = err_filled_data.reshape(1, 1, 1, err_filled_data.shape[0], err_filled_data.shape[1])
#         print(err_data.shape)
#         # 重塑之后，需要沿着数据维度进行堆叠
#         err_in = xr.Dataset({'err':(['model','date','time','lat','lon'],err_data),'mask':(['lat','lon'],mask_ocean_data)},coords={'time':[1],'lat':GLOBAL_LAT,'lon':GLOBAL_LON})
#         ds_all = xr.concat([ds_all, err_in], dim="date")
#         # print(ds_all)
#         # print(ds_all.err.shape)


# # # 存储2009年相关数据,作为训练集
# ds_all.to_netcdf("ds_train.nc")








# # 制作一个验证集
# # 循环得到每个nc文件的路径
# default_enso_dir = os.path.join(default_enso_dir, "2010")
# # print(default_enso_dir)
# paths = [os.path.join(default_enso_dir, f) for f in os.listdir(default_enso_dir) if f.endswith('.nc')]
#     # print(paths)  
# path_number = len(paths)
# # print(paths)
# # print(path_number)





# # 在循环的过程中制作相关的训练输入数据和输出数据
# # 在这里默认将数据全部置为nan，除非得到第一个数据
# ds_all = xr.Dataset()



# for path in paths:
#     ds = xr.open_dataset(path)


#     # 得到未处理nan值的数据
#     err_no_fillnan_data = ds.variables['err'].values
#     # 选择time = 0 的数据
#     err_time_data =  err_no_fillnan_data[0,:]
#     # 进行0值填充
#     err_filled_data = np.nan_to_num(err_time_data,nan = 0)
#     print(err_filled_data)
#     print(err_filled_data.shape)
#     # 需要对与数据进行reshape，将[625,721] -> [1,样本数,lev(1)，lat(625),lon(721)]
#     err_data = err_filled_data.reshape( 1, 1, err_filled_data.shape[0], err_filled_data.shape[1])
#     print(err_data.shape)
#     # 重塑之后，需要沿着数据维度进行堆叠
#     err_in = xr.Dataset({'err':(['date','time','lat','lon'],err_data),'mask':(['lat','lon'],mask_ocean_data)},coords={'time':[1],'lat':GLOBAL_LAT,'lon':GLOBAL_LON})
#     ds_all = xr.concat([ds_all, err_in], dim="date")
#     print(ds_all)
#     print(ds_all.err.shape)




# # # 得到一个ds_all(2010 年的数据制作验证集)




# #  读取err数据
# err_data = ds_all.variables['err'].values
# # print(err_data)
# # print(err_data.shape)
# # 同样获取其他的属性数据

# time = ds_all.variables['time'].values
# lat = ds_all.variables['lat'].values
# lon = ds_all.variables['lon'].values




# # 进行切分堆叠，需要将原始数据进行循环切分为输入历史数据，预测未来数据
# input_length = 3
# output_length = 3


# st_min = input_length - 1
# ed_max =err_data.shape[0] -output_length

# # 定义一个样本数，总样本-过去的时间长度-未来的预测时间长度


# # 定义一个符合测试集标准的高维矩阵，err_in 和 err_out，之后进行循环赋值即可。



# # 制作输入历史数据,不存在nino3.4数据
# ds_in = err_data[1 : 4]  # [12,4,625,721]
# print(ds_in.shape)

# ds_out = err_data[4 : 7]
# print(ds_out.shape)
# # 首先定义一个样本
# err_test = xr.Dataset({'err_in': (['input_length','time','lat','lon'],ds_in),
#                             'err_out': (['output_length','time','lat','lon'],ds_out),'mask':(['lat','lon'],mask_ocean_data)},coords={'time':time,'lat':lat,'lon':lon})


# # 直接进行一个循环堆叠
# for i in range(err_data.shape[0]):
#     rd = random.randint(st_min+1, ed_max - 1)
#     # 制作输入历史数据,不存在nino3.4数据
#     ds_in = err_data[rd - input_length + 1 : rd + 1]  # [12,4,625,721]
#     # 制作输出预测数据,同样是范围数据,不存在nino3.4数据ll the input arrays must have same number 
#     ds_out = err_data[rd + 1 : rd + output_length + 1]
    
#     err_temp_data = xr.Dataset({'err_in':(['input_length','time','lat','lon'],ds_in),
#                             'err_out':(['output_length','time','lat','lon'],ds_out),'mask':(['lat','lon'],mask_ocean_data)
#                             },coords={'time':time,'lat':lat,'lon':lon})
#     err_test = xr.concat([err_test,err_temp_data],dim='ngroup')  


# print(err_test)
# err_test.to_netcdf("ds_test.nc")






# # # 获取数据地址
# _CURR_DIR = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))


# cfg = OmegaConf.create()
# cfg.root_dir = os.path.abspath(os.path.realpath(os.path.join(_CURR_DIR, "..")))
# cfg.datasets_dir =os.path.join(cfg.root_dir, "data")




# # 得到默认的数据地址
# default_enso_dir =cfg.datasets_dir
# print(default_enso_dir)

# # 读取得到相关的lat，lon信息
# default_tmp_dir = os.path.join(default_enso_dir, "2009\\20090326.nc")
# # print(default_tmp_dir)
# tmp_data = xr.open_dataset(default_tmp_dir)
# print(tmp_data)
# GLOBAL_LAT = tmp_data.variables['lat'].values
# GLOBAL_LON = tmp_data.variables['lon'].values
# err_data = tmp_data['err']
# # 选择time = 0 的数据
# err_in_data =  err_data[0,:]
# # 进行0值填充
# err_filled_in_data = np.nan_to_num(err_in_data,nan = 0)
# # 选择 time 后续的数据
# err_out_data =  err_data[1:,:]
# err_filled_out_data = np.nan_to_num(err_out_data,nan = 0)
# # 需要对与数据进行reshape，将[625,721] -> [1,样本数,lev(1)，lat(625),lon(721)]
# err_in_data = err_filled_in_data.reshape(1, 1, 1, err_filled_in_data.shape[0], err_filled_in_data.shape[1])
# err_out_data = err_filled_out_data.reshape(1, err_filled_out_data.shape[0], 1, err_filled_out_data.shape[1], err_filled_out_data.shape[2])
# # 重塑之后，需要沿着数据维度进行堆叠
# ds_all = xr.Dataset({'err_in':(['ngroup','input_length','time','lat','lon'],err_in_data),
#                      'err_out':(['ngroup','output_length','time','lat','lon'],err_out_data)},coords={'time':[1],'lat':GLOBAL_LAT,'lon':GLOBAL_LON})


# years = ["2009","2010","2011","2012","2013","2014"]
# for year in years:
#     default_err_dir = os.path.join(default_enso_dir, year)
#     # print(default_enso_dir)
#     paths = [os.path.join(default_err_dir, f) for f in os.listdir(default_err_dir) if f.endswith('.nc')]
#         # print(paths)  
#     path_number = len(paths)
#     print(paths)
#     print(path_number)
#     for path in paths:
#         ds = xr.open_dataset(path)
#         # 得到未处理nan值的数据
#         err_no_fillnan_data = ds.variables['err'].values
#         # 选择time = 0 的数据
#         err_in_data =  err_no_fillnan_data[0,:]
#         # 进行0值填充
#         err_filled_in_data = np.nan_to_num(err_in_data,nan = 0)
#         # 选择 time 后续的数据
#         err_out_data =  err_no_fillnan_data[1:,:]
#         err_filled_out_data = np.nan_to_num(err_out_data,nan = 0)
#         # 需要对与数据进行reshape，将[625,721] -> [1,样本数,lev(1)，lat(625),lon(721)]
#         err_in_data = err_filled_in_data.reshape(1, 1, 1, err_filled_in_data.shape[0], err_filled_in_data.shape[1])
#         err_out_data = err_filled_out_data.reshape(1, err_filled_out_data.shape[0], 1, err_filled_in_data.shape[1], err_filled_in_data.shape[2])
#         # 重塑之后，需要沿着数据维度进行堆叠
#         err_in = xr.Dataset({'err_in':(['ngroup','input_length','time','lat','lon'],err_in_data),
#                              'err_out':(['ngroup','output_length','time','lat','lon'],err_out_data)},coords={'time':[1],'lat':GLOBAL_LAT,'lon':GLOBAL_LON})

#         ds_all = xr.concat([ds_all, err_in], dim="ngroup")


# ds_all.to_netcdf("ds_train.nc")






class make_dataset2(IterableDataset):
    """
    online reading dataset
    """
    # 初始化读入nc文件的数据
    def __init__(self, mypara):
        self.mypara = mypara
        print(mypara.adr_pretr)




        data_in = xr.open_dataset(mypara.adr_pretr)
        self.time = data_in["time"].values
        self.lat = data_in["lat"].values
        # print(self.lat)
        self.lon = data_in["lon"].values
        # print(self.lon)
        self.time_range = mypara.time_range
        self.lon_range = mypara.lon_range
        self.lat_range = mypara.lat_range
        self.input_length = mypara.input_length
        self.output_length = mypara.output_length
        self.all_group = mypara.all_group
        # 获取nc文件中的海温场数据
        temp = data_in["err"][
            :,
            :,
            mypara.time_range[0] : mypara.time_range[1],
            mypara.lat_range[0] : mypara.lat_range[1],
            mypara.lon_range[0] : mypara.lon_range[1],
        ].values

        # temp = data_in["err"].values
        # 将temp中的所有NaN值替换为0。
        temp = np.nan_to_num(temp)
        # 将temp中的所有大于999的数据值替换为0,本质上是对于数据过滤
        temp[abs(temp) > 999] = 0
        # 需要在这里将数据进行最大最小归一化
        # 首先定义一个最大最小化函数
        # print(temp.shape)
        # MinMaxScaler = MinMaxScaler()
        # temp_normalized = MinMaxScaler.fit_transform(temp)


        # 判断是否需要考虑风应力因素,其中taux表示为经向风应力,tauy表示为纬向风应力
        if mypara.needtauxy:
            print("loading tauxy...")
            taux = data_in["tauxNor"][
                :,
                :,
                mypara.lat_range[0] : mypara.lat_range[1],
                mypara.lon_range[0] : mypara.lon_range[1],
            ].values
            taux = np.nan_to_num(taux)
            taux[abs(taux) > 999] = 0
            tauy = data_in["tauyNor"][
                :,
                :,
                mypara.lat_range[0] : mypara.lat_range[1],
                mypara.lon_range[0] : mypara.lon_range[1],
            ].values
            tauy = np.nan_to_num(tauy)
            tauy[abs(tauy) > 999] = 0
            # 将taux,tauy,tmp数据进行拼接,沿着第三个维度进行拼接
            self.field_data = np.concatenate(
                (taux[:, :, None], tauy[:, :, None], temp), axis=2
            )
            del temp, taux, tauy
        else:
            # 并未考虑风应力因素的影响,单纯只考虑海温场的数据

            self.field_data = temp
            # print(self.field_data)
            # print(self.field_data.shape)
            del temp

    def __iter__(self):
        # 设由于self。input_length = 12 ,所以可以得出st_min = 11
        st_min = self.input_length - 1
        # print(st_min)
        # 同理由于该self.field_data.shape[1] = 12，而self.output_length =20，故ed_max = -8 
        ed_max = self.field_data.shape[1] - self.output_length
        # print(ed_max)
        # 在所有的输入数据模式中，在这里制作一个关于输入数据喝输出数据的形式
        for i in range(self.all_group):
            # 随机产生一个0-样本数之间的整数，选择输入数据集中的第一个维度数据
            rd_m = random.randint(0, self.field_data.shape[0] - 1)
            # 随机产生一个从随机整数
            rd = random.randint(st_min, ed_max - 1)
            # 制作输入历史数据,不存在nino3.4数据
            dataX = self.field_data[rd_m, rd - self.input_length + 1 : rd + 1]
            # print(dataX)
            # print(dataX.shape)
            # 制作输出预测数据,同样是范围数据,不存在nino3.4数据
            dataY = self.field_data[rd_m, rd + 1 : rd + self.output_length + 1]
            # print(dataY)
            # print(dataY.shape)
            yield dataX, dataY

    def selectregion(self):
        return {
            "lon: {}E to {}E".format(
                self.lon[self.lon_range[0]],
                self.lon[self.lon_range[1] - 1],
            ),
            "lat: {}N to {}N".format(
                self.lat[self.lat_range[0]],
                self.lat[self.lat_range[1] - 1],
            ),
            "temp time: {}m to {}m".format(
                self.time[self.time_range[0]], self.time[self.time_range[1] - 1]
            ),
        }




class make_testdataset(Dataset):
    def __init__(self, mypara, ngroup):
        self.mypara = mypara
        data_in = xr.open_dataset(mypara.adr_eval)
        self.time = data_in["time"].values
        self.lat = data_in["lat"].values
        self.lon = data_in["lon"].values
        self.time_range = mypara.time_range
        self.lon_range = mypara.lon_range
        self.lat_range = mypara.lat_range
        # 读入历史输入数据，海洋温度异常场数据
        temp_in = data_in["err_in"][
            :,
            :,
            mypara.time_range[0] : mypara.time_range[1],
            mypara.lat_range[0] : mypara.lat_range[1],
            mypara.lon_range[0] : mypara.lon_range[1],
        ].values
        temp_in = np.nan_to_num(temp_in)
        temp_in[abs(temp_in) > 999] = 0
        assert mypara.input_length == temp_in.shape[1]
        if mypara.needtauxy:
            print("loading tauxy...")
            taux_in = data_in["tauxNor_in"][
                :,
                :,
                mypara.lat_range[0] : mypara.lat_range[1],
                mypara.lon_range[0] : mypara.lon_range[1],
            ].values
            taux_in = np.nan_to_num(taux_in)
            taux_in[abs(taux_in) > 999] = 0
            tauy_in = data_in["tauyNor_in"][
                :,
                :,
                mypara.lat_range[0] : mypara.lat_range[1],
                mypara.lon_range[0] : mypara.lon_range[1],
            ].values
            tauy_in = np.nan_to_num(tauy_in)
            tauy_in[abs(tauy_in) > 999] = 0
            field_data_in = np.concatenate(
                (taux_in[:, :, None], tauy_in[:, :, None], temp_in), axis=2
            )
            del temp_in, taux_in, tauy_in
        else:
            field_data_in = temp_in
            del temp_in
        # ====================out
        # 读取未来预测数据，海洋温度异常场数据
        temp_out = data_in["err_out"][
            :,
            :,
            mypara.time_range[0] : mypara.time_range[1],
            mypara.lat_range[0] : mypara.lat_range[1],
            mypara.lon_range[0] : mypara.lon_range[1],
        ].values
        temp_out = np.nan_to_num(temp_out)
        temp_out[abs(temp_out) > 999] = 0
        assert mypara.output_length == temp_out.shape[1]
        if mypara.needtauxy:
            print("loading tauxy...")
            taux_out = data_in["tauxNor_out"][
                :,
                :,
                mypara.lat_range[0] : mypara.lat_range[1],
                mypara.lon_range[0] : mypara.lon_range[1],
            ].values
            taux_out = np.nan_to_num(taux_out)
            taux_out[abs(taux_out) > 999] = 0
            tauy_out = data_in["tauyNor_out"][
                :,
                :,
                mypara.lat_range[0] : mypara.lat_range[1],
                mypara.lon_range[0] : mypara.lon_range[1],
            ].values
            tauy_out = np.nan_to_num(tauy_out)
            tauy_out[abs(tauy_out) > 999] = 0
            # ------------
            field_data_out = np.concatenate(
                (taux_out[:, :, None], tauy_out[:, :, None], temp_out), axis=2
            )
            del temp_out, taux_out, tauy_out
        else:
            field_data_out = temp_out
            del temp_out
        # -----------------------------
        self.dataX, self.dataY = self.deal_testdata(
            field_data_in=field_data_in, field_data_out=field_data_out, ngroup=ngroup
        )
        del field_data_in, field_data_out

    def deal_testdata(self, field_data_in, field_data_out, ngroup):
        print("Random sampling...")
        # 历史时间长度，lb = 12
        lb = field_data_in.shape[1]
        # 输出预测时间长度，output_length = 20
        output_length = field_data_out.shape[1]
        # 初始化一个多维数组，形状为数组group数，lb历史时间长度，以及其他的形状长度
        out_field_x = np.zeros(
            [
                ngroup,
                lb,
                field_data_in.shape[2],
                field_data_in.shape[3],
                field_data_in.shape[4],
            ]
        )
        # 同理这也是多维数组，输出的多维数组数据的形状
        out_field_y = np.zeros(
            [
                ngroup,
                output_length,
                field_data_out.shape[2],
                field_data_out.shape[3],
                field_data_out.shape[4],
            ]
        )

        # 以下代码是建立起输入历史数据（12时间长度），和输出预测数据（20时间长度）
        iii = 0
        for j in range(ngroup):
            rd = random.randint(0, field_data_in.shape[0] - 1)
            out_field_x[iii] = field_data_in[rd]
            out_field_y[iii] = field_data_out[rd]
            iii += 1
        print("End of sampling...")
        return out_field_x, out_field_y

    def getdatashape(self):
        return {
            "dataX": self.dataX.shape,
            "dataY": self.dataY.shape,
        }

    def selectregion(self):
        return {
            "lon: {}E to {}E".format(
                self.lon[self.lon_range[0]],
                self.lon[self.lon_range[1] - 1],
            ),
            "lat: {}S to {}N".format(
                self.lat[self.lat_range[0]],
                self.lat[self.lat_range[1] - 1],
            ),
            "time: {}m to {}m".format(
                self.time[self.time_range[0]], self.time[self.time_range[1] - 1]
            ),
        }

    def __len__(self):
        return self.dataX.shape[0]

    def __getitem__(self, idx):
        return self.dataX[idx], self.dataY[idx]



class make_TFdataset(Dataset):
    def __init__(
        self,
        address,
        mypara,
        ngroup=None,
    ):
        self.mypara = mypara
        print(address)
        data_in = xr.open_dataset(address)
        self.lev = data_in["lev"].values
        self.lat = data_in["lat"].values
        self.lon = data_in["lon"].values
        self.lev_range = mypara.lev_range
        self.lon_range = mypara.lon_range
        self.lat_range = mypara.lat_range

        temp_in = data_in["err"][
            :,
            :,
            self.lev_range[0] : self.lev_range[1],
            self.lat_range[0] : self.lat_range[1],
            self.lon_range[0] : self.lon_range[1],
        ].values
        temp_in = np.nan_to_num(temp_in)
        temp_in[abs(temp_in) > 999] = 0
        assert mypara.input_length == temp_in.shape[1]
        if mypara.needtauxy:
            print("loading tauxy...")
            taux_in = data_in["tauxNor_in"][
                :,
                :,
                self.lat_range[0] : self.lat_range[1],
                self.lon_range[0] : self.lon_range[1],
            ].values
            taux_in = np.nan_to_num(taux_in)
            taux_in[abs(taux_in) > 999] = 0
            tauy_in = data_in["tauyNor_in"][
                :,
                :,
                self.lat_range[0] : self.lat_range[1],
                self.lon_range[0] : self.lon_range[1],
            ].values
            tauy_in = np.nan_to_num(tauy_in)
            tauy_in[abs(tauy_in) > 999] = 0
            field_data_in = np.concatenate(
                (taux_in[:, :, None], tauy_in[:, :, None], temp_in), axis=2
            )
            del temp_in, taux_in, tauy_in
        else:
            field_data_in = temp_in
            del temp_in
        # ================
        temp_out = data_in["err"][
            :,
            :,
            self.lev_range[0] : self.lev_range[1],
            self.lat_range[0] : self.lat_range[1],
            self.lon_range[0] : self.lon_range[1],
        ].values
        temp_out = np.nan_to_num(temp_out)
        temp_out[abs(temp_out) > 999] = 0
        assert mypara.output_length == temp_out.shape[1]
        if mypara.needtauxy:
            print("loading tauxy...")
            taux_out = data_in["tauxNor_out"][
                :,
                :,
                self.lat_range[0] : self.lat_range[1],
                self.lon_range[0] : self.lon_range[1],
            ].values
            taux_out = np.nan_to_num(taux_out)
            taux_out[abs(taux_out) > 999] = 0
            tauy_out = data_in["tauyNor_out"][
                :,
                :,
                self.lat_range[0] : self.lat_range[1],
                self.lon_range[0] : self.lon_range[1],
            ].values
            tauy_out = np.nan_to_num(tauy_out)
            tauy_out[abs(tauy_out) > 999] = 0
            # ------------
            field_data_out = np.concatenate(
                (taux_out[:, :, None], tauy_out[:, :, None], temp_out), axis=2
            )  # [group,lb,all_lev,lat,lon]
            del temp_out, taux_out, tauy_out
        else:
            field_data_out = temp_out
            del temp_out
        # -----------------------------
        self.dataX, self.dataY = self.deal_testdata(
            field_data_in=field_data_in, field_data_out=field_data_out, ngroup=ngroup
        )
        del field_data_in, field_data_out

    def deal_testdata(self, field_data_in, field_data_out, ngroup):
        lb = field_data_in.shape[1]
        output_length = field_data_out.shape[1]
        if ngroup is None:
            ngroup = field_data_in.shape[0]
        out_field_x = np.zeros(
            [
                ngroup,
                lb,
                field_data_in.shape[2],
                field_data_in.shape[3],
                field_data_in.shape[4],
            ]
        )
        out_field_y = np.zeros(
            [
                ngroup,
                output_length,
                field_data_out.shape[2],
                field_data_out.shape[3],
                field_data_out.shape[4],
            ]
        )
        iii = 0
        for j in range(ngroup):
            rd = random.randint(0, field_data_in.shape[0] - 1)
            out_field_x[iii] = field_data_in[rd]
            out_field_y[iii] = field_data_out[rd]
            iii += 1
        return out_field_x, out_field_y

    def getdatashape(self):
        return {
            "dataX": self.dataX.shape,
            "dataY": self.dataY.shape,
        }

    def selectregion(self):
        return {
            "lon: {}E to {}E".format(
                self.lon[self.lon_range[0]],
                self.lon[self.lon_range[1] - 1],
            ),
            "lat: {}N to {}N".format(
                self.lat[self.lat_range[0]],
                self.lat[self.lat_range[1] - 1],
            ),
            "lev: {}m to {}m".format(
                self.lev[self.lev_range[0]], self.lev[self.lev_range[1] - 1]
            ),
        }

    def __len__(self):
        return self.dataX.shape[0]

    def __getitem__(self, idx):
        return self.dataX[idx], self.dataY[idx]








# 这一部分是编写最终制作数据集的部分.需要进行详细的特殊处理.

class make_All_dataset(Dataset):
    def __init__(
        self,
        address,
        mypara,
    ):
        self.mypara = mypara
        self.address = address
        default_temp_dir = os.path.join(address, "temp/")
        default_err_dir = os.path.join(self.address,"err/")
        folder_names = sorted([entry.name for entry in os.scandir(default_temp_dir) if entry.is_dir()])
        self.folder_names = folder_names
        self.default_temp_dir = default_temp_dir
        self.default_err_dir = default_err_dir

    def __len__(self):
        return len(self.folder_names)-3   
            
    def __getitem__(self, index):

        actual_index = index + 3

        # 生成数据并返回
        dataX, dataY = self._generate_data(actual_index)

        return dataX, dataY
    
    def _generate_data(self, index):
        #  获取得到前两天的子文件路径名，前一天的子文件路径名，同样需要获取得到对应路径下的子文件。
        # 当前的子文件夹中时间维度上数据缺失，即缺少24小时的模式预报数据，需要复制前一天的48小时模式预报数据，或者是前两天的72小时模式预报数据
        # 获取前一天的文件路径
        default_temp_dir = os.path.join(self.address, "temp/")
        default_err_dir = os.path.join(self.address, "err/")

        prev_sub_dir_1 = os.path.join(default_temp_dir,self.folder_names[index-1])

        # 需要获取对应子文件夹下的数据列表
        pre_1_temp_paths = [os.path.join(prev_sub_dir_1, f) for f in sorted(os.listdir(prev_sub_dir_1)) if f.endswith('.nc')]
        # 获取前两天的文件路径
        prev_sub_dir_2 = os.path.join(default_temp_dir,self.folder_names[index-2])
        # 获取对应子文件夹下的数据列表
        pre_2_temp_paths = [os.path.join(prev_sub_dir_2, f) for f in sorted(os.listdir(prev_sub_dir_2)) if f.endswith('.nc')]

        # 获取前三天的文件路径
        prev_sub_dir_3 = os.path.join(default_temp_dir,self.folder_names[index-3])
        # 获取对应子文件夹下的数据列表
        pre_3_temp_paths = [os.path.join(prev_sub_dir_3, f) for f in sorted(os.listdir(prev_sub_dir_3)) if f.endswith('.nc')]
        # 获取每个子文件路径的前24小时的路径，整合为每个循环所得到的总体路径，将数据堆叠之后，得到模型输入的样例。
        temp_paths=np.concatenate((pre_3_temp_paths[:8],pre_2_temp_paths[:8],pre_1_temp_paths[:9]),axis=0)
        

        # 用于存储堆叠好的模型输入数据
        stacked_temp_data = None

        # 定义归一化函数
        min_max_normalization = MinMaxScaler()

        # 循环读取数据制作为模型输入数据
        for path in temp_paths:
            # 由于路径都是选择好的，直接读取数据就行
            temp_nc_data  = xr.open_dataset(path)
            # 测试是否读取到了数据
            # print(temp_nc_data)
            temp_data = temp_nc_data['temp'].values

            # 在这里将每个时间间隔的数据中的NaN值进行替换，简单使用零值替换
            temp_data = np.nan_to_num(temp_data)
            temp_data[abs(temp_data) > 999] = 0
            
            
            # 将三维数据转变为二维数据
            num_samples, num_rows, num_cols = temp_data.shape
            reshaped_data = temp_data.reshape(num_samples * num_rows ,  num_cols)
            # 在这进行归一化
            temp_nor_data = min_max_normalization.fit_transform(reshaped_data)

            # 重新转为三维数据
            temp_data = temp_nor_data.reshape(num_samples, num_rows, num_cols)


            if stacked_temp_data is None:
                # 如果是第一个数组，则直接赋值给 stacked_temp_data，是否需要进行重塑
                stacked_temp_data = temp_data.reshape(1,temp_data.shape[0],temp_data.shape[1],temp_data.shape[2])
                # print(stacked_temp_data.shape)
            else:
                # 使用 np.concatenate() 函数将当前数组与 stacked_temp_data 进行连接
                stacked_temp_data = np.concatenate((stacked_temp_data, temp_data.reshape(1,temp_data.shape[0],temp_data.shape[1],temp_data.shape[2])), axis=0)
        # 成功获取得到模型输入数据
        
        assert stacked_temp_data.shape[0] == 25
        # 现在需要获取得到模型的输出数据

        # 这里用来获取得到误差数据路径,以及测试是否正确获取得到相关的路径.
        default_err_sub_dir = os.path.join(default_err_dir,self.folder_names[index])
        print(default_err_sub_dir)
        # 同样在这里获取关于模型的输出数据路径，即获取得到err文件路径
        err_paths = [os.path.join(default_err_sub_dir, f) for f in os.listdir(default_err_sub_dir) if f.endswith('.nc')]

        stacked_err_data = None
        # 在这里制作输出数据集
        for path in  err_paths:
            # 循环读取误差数据子文件夹的72小时误差数据，从而制作为输出数据集。
            err_nc_data = xr.open_dataset(path)
            # 测试是否读取到数据
            # print(err_nc_data)
            # 获取真实的误差数据
            err_data = err_nc_data['sst'].values
            # 将NaN值进行替换
            err_data = np.nan_to_num(err_data)
            err_data[abs(err_data) > 999] = 0
            # 将err子文件中的误差数据进行堆叠，从而实现从当前时刻72小时误差数据的整合
            if stacked_err_data is None:
                stacked_err_data = err_data.reshape(1,err_data.shape[0],err_data.shape[1],err_data.shape[2])
            else:
                stacked_err_data = np.concatenate((stacked_err_data,err_data.reshape(1,err_data.shape[0],err_data.shape[1],err_data.shape[2])),axis=0)
        
        # 需要判断长度是否满足需求
        assert stacked_err_data.shape[0] ==25

        # 在这里进行反馈,使用yield 进行传递.
        dataX = stacked_temp_data
        dataY = stacked_err_data
        # del stacked_err_data,stacked_err_data
        return dataX,dataY   

    def selectregion(self):
        return {
            "lon: {}E to {}E".format(
                self.mypara.lon_range[0],
                self.mypara.lon_range[1],
            ),
            "lat: {}N to {}N".format(
                self.mypara.lat_range[0],
                self.mypara.lat_range[1],
            ),
        }



  