from Geoformer import Geoformer
from myconfig import mypara
import torch
from torch.utils.data import DataLoader
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
# from LoadData import make_dataset2, make_testdataset
from SST import make_dataset2, make_testdataset,make_TFdataset,make_All_dataset
from copy import deepcopy
import xarray as xr
# 定义学习率调整器，用于调整优化器中的学习率。
class lrwarm:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )
    

class modelTrainer:
    def __init__(self, mypara):
        # 判断输入和输出的数据通道是否相同
        assert mypara.input_channal == mypara.output_channal
        # 获取相关的配置参数
        self.mypara = mypara
        # 获取配置的GPU地址
        self.device = mypara.device
        # 获取设置的transformer模型,
        self.mymodel = Geoformer(mypara).to(mypara.device)
        # 设置优化器，采用adam 的优化器
        # self.adam = torch.optim.Adam(self.mymodel.parameters(), lr=5e-5)
        adam = torch.optim.Adam(self.mymodel.parameters(), lr=0)
        # 这行代码的作用是计算学习率的初始值，它通常用于优化算法中的梯度下降过程。这个学习率的初始值是基于一些参数的计算结果得出的。
        factor = math.sqrt(mypara.d_size * mypara.warmup) * 0.0015
        # 初始化一个优化器对象
        self.opt = lrwarm(mypara.d_size, factor, mypara.warmup, optimizer=adam)
        # 
        self.sstlevel = 0
        if self.mypara.needtauxy:
            self.sstlevel = 2
        # 创建一个张量，并将其发送到特定的设备中，设置一个nino的权重矩阵
        ninoweight = torch.from_numpy(
            np.array([1.5] * 4 + [2] * 7 + [3] * 8 + [4] * 6)
            * np.log(np.arange(25) + 1)
        ).to(mypara.device)
        self.ninoweight = ninoweight[: self.mypara.output_length]
        print(self.ninoweight)
        print(len(self.ninoweight))
        print(len(ninoweight))


    # 计算Nino的得分
    def calscore(self, y_pred, y_true):
        # compute Nino score
        with torch.no_grad():
            pred = y_pred - y_pred.mean(dim=0, keepdim=True)
            true = y_true - y_true.mean(dim=0, keepdim=True)
            cor = (pred * true).sum(dim=0) / (
                torch.sqrt(torch.sum(pred ** 2, dim=0) * torch.sum(true ** 2, dim=0))
                + 1e-6
            )
            acc = (self.ninoweight * cor).sum()
            rmse = torch.mean((y_pred - y_true) ** 2, dim=0).sqrt().sum()
            sc = 2 / 3.0 * acc - rmse
        return sc.item()
    
    # 设置为均方根误差为损失函数的一部分
    def loss_var(self, y_pred, y_true):
        mae = torch.mean(torch.abs(y_pred - y_true), dim=[3, 4])
        mae = mae.mean(dim=0)
        mae = torch.sum(mae, dim=[0, 1])
        return mae
    



    # 设置为nino3.4 的指数为损失函数的一部分
    def loss_nino(self, y_pred, y_true):
        # with torch.no_grad():
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2, dim=0))
        return rmse.sum()
    # 将两部分是损失函数结合起来
    def combien_loss(self, loss1, loss2):
        combine_loss = loss1 + loss2
        return combine_loss
    # 模型的预测部分
    def model_pred(self, dataloader):

        self.mymodel.eval()

        # nino_pred = []
        # var_pred = []
        # nino_true = []
        # var_true = []
        with torch.no_grad():
            for j, (input_var, var_true1)  in enumerate(dataloader):
                SST = var_true1[:, :, self.sstlevel]
                nino_true1 = SST[
                    :,
                    :,
                    self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[1],
                ].mean(dim=[2, 3])



                out_var = self.mymodel(
                    input_var.float().to(self.device),
                    predictand=None,
                    train=False,
                )
                SST_out = out_var[:, :, self.sstlevel]
                out_nino = SST_out[
                    :,
                    :,
                    self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[1],
                ].mean(dim=[2, 3])
                # var_true.append(var_true1)
                # nino_true.append(nino_true1)
                # var_pred.append(out_var)
                # nino_pred.append(out_nino)
                # ninosc = self.calscore(nino_pred, nino_true.float().to(self.device))
                loss_var = self.loss_var(out_var, var_true1.float().to(self.device)).item()
                print("-->Evaluation... \nloss_var:{:.3f} ".format(
                            loss_var
                        ))
                
                if j >= 2 :
                    break
                # loss_nino = self.loss_nino(
                #     nino_pred, nino_true.float().to(self.device)
                # ).item()


                
            # var_pred = torch.cat(var_pred, dim=0)
            # nino_pred = torch.cat(nino_pred, dim=0)
            # nino_true = torch.cat(nino_true, dim=0)
            # var_true = torch.cat(var_true, dim=0)
            # --------------------
            # ninosc = self.calscore(nino_pred, nino_true.float().to(self.device))
            # loss_var = self.loss_var(var_pred, var_true.float().to(self.device)).item()
            # loss_nino = self.loss_nino(
            #     nino_pred, nino_true.float().to(self.device)
            # ).item()

            # combine_loss = self.combien_loss(loss_var, loss_nino)
            # combine_loss = loss_var
        return (
            # var_pred,
            # nino_pred,
            loss_var,
            # loss_nino,
            # combine_loss,
            # ninosc,
        )
    # 模型的训练部分
    def train_model(self, dataset_train, dataset_eval):
        chk_path = self.mypara.model_savepath + "Geoformer.pkl"
        torch.manual_seed(self.mypara.seeds)
        # 加载训练数据
        dataloader_train = DataLoader(
            dataset_train, batch_size=self.mypara.batch_size_train, shuffle=True
        )
        # 加载评估数据
        dataloader_eval = DataLoader(
            dataset_eval, batch_size=self.mypara.batch_size_eval, shuffle=True
        )
        count = 0
        # 记录最小值，这里表示为最小值，负无穷，需要更改为最大值，即正无穷
        best = math.inf
        print(best)
        # 进行半监督学习的部分，首先将预测内容一部分进行替换为观测数据，初始化的半监督比例
        sv_ratio = 1
        for i_epoch in range(self.mypara.num_epochs):
            print("==========" * 8)
            print("\n-->epoch: {0}".format(i_epoch))
            # --------
            # -train
            # 开始训练
            self.mymodel.train()
            # 开始遍历数据
            for j, (input_var, var_true) in enumerate(dataloader_train):
                SST = var_true[:, :, self.sstlevel]
                nino_true = SST[
                    :,
                    :,
                    self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[1],
                ].mean(dim=[2, 3])
                if sv_ratio > 0:
                    sv_ratio = max(sv_ratio - 2.5e-4, 0)
                # -------training for one batch
                var_pred = self.mymodel(
                    input_var.float().to(self.device),
                    var_true.float().to(self.device),
                    train=True,
                    sv_ratio=sv_ratio,
                )
                SST_pred = var_pred[:, :, self.sstlevel]
                nino_pred = SST_pred[
                    :,
                    :,
                    self.mypara.lat_nino_relative[0] : self.mypara.lat_nino_relative[1],
                    self.mypara.lon_nino_relative[0] : self.mypara.lon_nino_relative[1],
                ].mean(dim=[2, 3])
                self.opt.optimizer.zero_grad()
                # self.adam.zero_grad(),获取得到相关的损失值
                loss_var = self.loss_var(var_pred, var_true.float().to(self.device))
                loss_nino = self.loss_nino(nino_pred, nino_true.float().to(self.device))
                score = self.calscore(nino_pred, nino_true.float().to(self.device))
                # loss_var.backward()
                # combine_loss = self.combien_loss(loss_var, loss_nino)


                # 将损失函数进行反向传播
                # 进行权重列表的设置
                combine_loss = (self.ninoweight * loss_var).sum()
                # print(combine_loss)
                combine_loss.backward()
                self.opt.step()
                # self.adam.step()

                # 获取得到训练集训练过程中的损失值，获取得到一个当前训练epoch中损失值最小的参数
                if j % 1 == 0:
                    print(
                        "\n-->batch:{} loss_var:{:.2f}, loss_nino:{:.2f}, score:{:.3f}".format(
                            j, loss_var, loss_nino, score
                        )
                    )
                # 通过下列的代码进行一个模型参数的存储
                # ---------Intensive verification,强化核查
                # 这段代码的作用是在训练模型的过程中，当训练次数 i_epoch 大于等于 4，且每训练 200 个样本后，获得验证集损失最小的模型参数
                if (i_epoch + 1 >= 1) and (j + 1) % 10 == 0:
                    (
                        # _,
                        # _,
                        lossvar_eval,
                        # lossnino_eval,
                        # comloss_eval,
                        # sceval,
                    ) = self.model_pred(dataloader=dataloader_eval)
                    print(
                        "-->Evaluation... \nloss_var:{:.3f} ".format(
                            lossvar_eval
                        )
                    )


                    # 保存使得模型参数得到最小的损失值，模型参数
                    if lossvar_eval < best:
                        torch.save(
                            self.mymodel.state_dict(),
                            chk_path,
                        )
                        best = lossvar_eval
                        count = 0
                        print("\nsaving model...  ")
                        print(best)
            # ----------after one epoch-----------，在进行训练一个epoch之后
                # 获取得到测试eval部分的相关损失值，通过测试集进行比较得到训练损失最小的模型参数。
            (
                # _,
                # _,
                lossvar_eval,
                # lossnino_eval,
                # comloss_eval,
                # sceval,
            ) = self.model_pred(dataloader=dataloader_eval)
            # 打印出相关的损失值
            print(
                "\n-->epoch{} end... \nloss_var:{:.3f}".format(
                    i_epoch, lossvar_eval
                    )
                )
            # 如果实际训练损失值小于最小的损失值时，记录count
            if lossvar_eval <= best:
                # 如何大于最小的损失值时，需要保存相关的模型参数，需要进行修改，使得代码能够保存最小损失值的模型参数
                count = 0
                print(
                    "\nsc is decrease from {:.3f} to {:.3f}   \nsaving model...\n".format(
                        best, lossvar_eval
                    )
                )
                # 记录损失最小的模型参数，优化器的状态信息等，需要进行修改
                torch.save(
                    self.mymodel.state_dict(),
                    chk_path,
                )
                best = lossvar_eval
                
            else:
                count += 1
                print("\nsc is not decrease for {} epoch".format(count))

            # ---------early stop,提前达到停止状态
            if count == self.mypara.patience:
                print(
                    "\n-----!!!early stopping reached, max(sceval)= {:3f}!!!-----".format(
                        best
                    )
                )
                break
        del self.mymodel


if __name__ == "__main__":



    print(mypara.__dict__)

    # 首先mypara，需要进行循环读入SST模式预报数据，故此需要一些定义训练数据的地址。
    
    # 由于包含太平洋和印度洋两个区域的数据，故需要单独将数据进行读入测试（暂定)。首先将太平洋的数据进行读取



    # 按照迁移学习的文件制作数据集,直接制作为相关的总体数据集,之后再进行切分,将总体的数据集切分为训练集,测试集,和验证集.

    # 首先定义一个相关的总体数据路径
    adr_origins  = ("E:/SST_DATA/pacific/")
    # 之后在数据集中进行区分,从而得到对应的temp 模式预报数据路径,err误差数据路径.
    

    data_All= make_All_dataset(
        address=adr_origins,
        mypara=mypara,
    )
   
    


      # 切分数据集
    train_size = int(0.9 * len(data_All))
    eval_size = len(data_All) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        data_All, [train_size, eval_size], generator=torch.Generator().manual_seed(0)
    )



    # 获取得到相关的训练数据集和测试数据集之后，输入模型进行训练
    # -------------------------------------------------------------
    # 定义一个模型  
    trainer = modelTrainer(mypara)
    # 进行模型训练
    trainer.train_model(
        dataset_train=train_dataset,
        dataset_eval=eval_dataset,
    )
    