import torch

class Mypara:
    def __init__(self):
        pass

mypara = Mypara()
mypara.device = torch.device("cuda:0")
mypara.batch_size_train = 4
mypara.batch_size_eval = 4
mypara.num_epochs = 10
mypara.TFnum_epochs = 20
mypara.TFlr = 1.5e-5
mypara.early_stopping = True
mypara.patience = 4
mypara.warmup = 3000


mypara.interval = 4
mypara.TraindataProportion = 0.9
mypara.all_group = 13000

# mypara.adr_pretr = (
#     "D:\AIProject\Code_of_3D-Geoformer\Data\ds_train.nc"
# )


# mypara.adr_eval = (
#     "D:\AIProject\Code_of_3D-Geoformer\Data\ds_test.nc"
# )


mypara.adr_pretr = (
    "D:\AIProject\Code_of_3D-Geoformer\Data\ds_train.nc"
)



# mypara.adr_eval = (
#     "D:\AIProject\Code_of_3D-Geoformer\Data\ds_test.nc"
# )



mypara.needtauxy = False

mypara.input_channal = 1  # n_lev of 3D temperature
# 输出的通道数
mypara.output_channal = 1
# 输入数据的历史时间长度,注意训练的长度是3，预测是输入长度是12
mypara.input_length = 25
# 输出的历史时间长度，注意训练的长度是3，预测是输入长度是12
mypara.output_length = 25
# 数据的层数
mypara.lev_range = (0, 1)
mypara.time_range = (0, 1)
# 数据的经纬度范围,
# mypara.lon_range = (45, 165)
# mypara.lat_range = (0, 51)
mypara.lon_range = (98,175)
mypara.lat_range = (-20, 48)

# nino34 region,z，这里的数据是不需要的
mypara.lon_nino_relative = (49, 75)
mypara.lat_nino_relative = (15, 36)
# patch size，定义patch_size 的大小


mypara.patch_size = (20, 20)


# 在这里实际上需要去处理得到emb_spatial_size,可以使用网格数来进行修改定义，设置经度上的网格数，纬度上的网格数，从而来确定emb_spatial_szie
# 之后经纬度范围唯一的作用就是显示出真实经纬度区域
mypara.lat_grids_range = (0, 770)
mypara.lon_grids_range = (0,680)


# 需要在这里进行处理分辨率的问题，这个在维度上进行三次切分，经度上进in行四次切分，如果采用[625,721]的话，
# 计算输入数据在经度方向上的 patch 数量
mypara.H0 = int((mypara.lat_grids_range[1] - mypara.lat_grids_range[0]) / mypara.patch_size[0])
# 同样是计算输入数在维度方向上的patch数量
mypara.W0 = int((mypara.lon_grids_range[1] - mypara.lon_grids_range[0]) / mypara.patch_size[1])
# 计算得到输入数据经过patchEmbedding得到的空间嵌入向量长度
mypara.emb_spatial_size = mypara.H0 * mypara.W0

# 在这里进行设置数量。抽象将其进行多尺度融合
# model
mypara.model_savepath = "D:/AIProject/Code_of_3D-Geoformer/Data/tmp_module/"
mypara.seeds = 1
mypara.d_size = 256
mypara.nheads = 4
mypara.dim_feedforward = 512
mypara.dropout = 0.2
mypara.num_encoder_layers = 4
mypara.num_decoder_layers = 4
