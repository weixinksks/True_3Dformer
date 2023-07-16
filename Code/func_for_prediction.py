from Geoformer import Geoformer
import torch
from torch.utils.data import DataLoader
import numpy as np
import xarray as xr
from torch.utils.data import Dataset
import os
import random
from sklearn.preprocessing import MinMaxScaler
# 制作用于生成预测结果的模型输入数据，通过采用记录的模型参数来实现对于输入的预测结果生成
class make_dataset_test(Dataset):
    def __init__(
        self,
        address,
        needtauxy,
        time_range=(0, 1),
        lon_range=(0, 1),
        lat_range=(0, 1),
    ):
        data_in = xr.open_dataset(address)
        self.time = data_in["time"].values
        self.lat = data_in["lat"].values
        self.lon = data_in["lon"].values
        self.time_range = time_range
        self.lon_range = lon_range
        self.lat_range = lat_range

        temp = data_in["err_in"][
            :,
            :,
            time_range[0] : time_range[1],
            lat_range[0] : lat_range[1],
            lon_range[0] : lon_range[1],
        ].values
        temp = np.nan_to_num(temp)
        temp[abs(temp) > 999] = 0
        if needtauxy:
            taux = data_in["tauxNor"][
                :,
                :,
                lat_range[0] : lat_range[1],
                lon_range[0] : lon_range[1],
            ].values
            taux = np.nan_to_num(taux)
            taux[abs(taux) > 999] = 0
            tauy = data_in["tauyNor"][
                :,
                :,
                lat_range[0] : lat_range[1],
                lon_range[0] : lon_range[1],
            ].values
            tauy = np.nan_to_num(tauy)
            tauy[abs(tauy) > 999] = 0
            # --------------
            self.dataX = np.concatenate(
                (taux[:, :, None], tauy[:, :, None], temp), axis=2
            )
            del temp, taux, tauy
        else:
            self.dataX = temp
            del temp

    def getdatashape(self):
        return {
            "dataX.shape": self.dataX.shape,
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
            "time: {}m to {}m".format(
                self.time[self.time_range[0]], self.time[self.time_range[1] - 1]
            ),
        }

    def __len__(self):
        return self.dataX.shape[0]

    def __getitem__(self, idx):
        return self.dataX[idx]







# 这一部分是编写最终制作数据集的部分.需要进行详细的特殊处理.

def generate_data(default_temp_dir,default_err_dir):
    # 获取得到前两天的子文件路径名，前一天的子文件路径名，同样需要获取得到对应路径下的子文件。
    # 当前的子文件夹中时间维度上数据缺失，即缺少24小时的模式预报数据，需要复制前一天的48小时模式预报数据，或者是前两天的72小时模式预报数据
    # 获取前一天的文件路径


    folder_names = sorted([entry.name for entry in os.scandir(default_temp_dir) if entry.is_dir()])

    index = random.randint(3, len(folder_names))


    prev_sub_dir_1 = os.path.join(default_temp_dir,folder_names[index-1])

    # 需要获取对应子文件夹下的数据列表
    pre_1_temp_paths = [os.path.join(prev_sub_dir_1, f) for f in sorted(os.listdir(prev_sub_dir_1)) if f.endswith('.nc')]
    # 获取前两天的文件路径
    prev_sub_dir_2 = os.path.join(default_temp_dir,folder_names[index-2])
    # 获取对应子文件夹下的数据列表
    pre_2_temp_paths = [os.path.join(prev_sub_dir_2, f) for f in sorted(os.listdir(prev_sub_dir_2)) if f.endswith('.nc')]

    # 获取前三天的文件路径
    prev_sub_dir_3 = os.path.join(default_temp_dir,folder_names[index-3])
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
    default_err_sub_dir = os.path.join(default_err_dir,folder_names[index])
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
    dataX = stacked_temp_data.reshape(1,stacked_temp_data.shape[0],stacked_temp_data.shape[1],stacked_temp_data.shape[2],stacked_temp_data.shape[3])
    dataY = stacked_err_data.reshape(1,stacked_err_data.shape[0],stacked_err_data.shape[1],stacked_err_data.shape[2],stacked_err_data.shape[3])
    # del stacked_err_data,stacked_err_data
    return dataX,dataY   







# 定义一个预测函数，作用是对于测试数据进行预测
def func_pre(mypara, adr_model, adr_data):


    # 随机生成对应的文件子路径,从而实现随机选取不同时间节点的文件.
    # 在这里生成不同的路径，在生成数据的函数中随机定义一个下标，之后随机生成输入数据和输出数据
    default_temp_dir = os.path.join(adr_data, "temp/")
    default_err_dir = os.path.join(adr_data, "err/")
    
    print(default_temp_dir)
    print(default_err_dir)

    dataX,dataY = generate_data(default_temp_dir,default_err_dir)
    

    print(dataX)
    print(dataX.shape)
    print(dataY)
    print(dataY.shape)

 



    # -------------
    # 得到测试集的长度（记录测试集中输入模型的具体），同时打印测试集数据的形状
    test_group = len(dataX)
    print(test_group)

    # 制作输入样本（加上了batch_size信息）
    dataloader_test = DataLoader(
        dataX, batch_size=mypara.batch_size_eval, shuffle=False
    )


    print(dataloader_test)
    # 加载一个构建的模型
    mymodel = Geoformer(mypara).to(mypara.device)
    # 将保存好的损失最小的模型参数加载到设定的mymodel中，这个过程会使得mymodel对象的参数与保存的模型参数完全一致，可以直接用于推断或微调等任务
    mymodel.load_state_dict(torch.load(adr_model))
    # 将模型切换到评估模式下，关闭dropout等操作
    mymodel.eval()
 
    # 设置var_pred 用来记录输入的测试集数据所产生的预测结果，首先得到置为0的初始矩阵
    var_pred = np.zeros(
        [
            test_group,
            mypara.output_length,
            mypara.time_range[1]-mypara.time_range[0],
            mypara.lon_grids_range[1] - mypara.lon_grids_range[0],
            mypara.lat_grids_range[1] - mypara.lat_grids_range[0],
        ]
    )
    # 设置相关的索引
    ii = 0
    iii = 0
    # 使用训练好的模型进行推断，并将推断结果存储到var_pred中，表示该代码在不进行梯度计算的情况下执行
    with torch.no_grad():
        # 遍历dataloader_test测试中所有的输入数据
        for input_var in dataloader_test:
            # 将输入数据输入到模型中进行推断，得到与之匹配的预测输出数据（batch_size,outputdimension）
            out_var = mymodel(
                input_var.float().to(mypara.device),
                predictand=None,
                train=False,
            )
            # print(out_var)
            # 在每次推断结束后，将out_var的批次大小（即batch size）加到ii变量中
            ii += out_var.shape[0]
            if torch.cuda.is_available():
                # 存储这一批次大小的输出预测数据到var_pred中
                var_pred[iii:ii] = out_var.cpu().detach().numpy()
            else:
                var_pred[iii:ii] = out_var.detach().numpy()
            iii = ii
    del (
        out_var,
        input_var,
    )
    del mymodel, dataloader_test

    # 用训练过的模型去输出预测结果
    print(var_pred)
    # 输出的结果长度是(31, 20, 1, 30, 26)，后续并不需要进行平滑操作
    print(var_pred.shape)

    return (
        var_pred,
        dataY
    )
