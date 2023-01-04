## 实验环境

Pycharm  Pytorch框架

实验所需要的依赖包：

re    random    tarfile  numpy   torch    sklearn  matplotlib 

## 数据集下载

本次实验所使用的的数据集是IDMB（Internet Movie Datavase)的电影评论数据，数据集已在压缩包中

## 运行方式

`python  net.py`

若需替换LSTM,GRU(已在model.py中定义好)需在以下代码中更换

![1672717207489](C:\Users\123\AppData\Roaming\Typora\typora-user-images\1672717207489.png)

更改LSTM还需要在以下代码中更改最后一个隐藏层的输出shape

![1672717326417](C:\Users\123\AppData\Roaming\Typora\typora-user-images\1672717326417.png)

## 实验结果

|  RNN   |  LSTM  |  GRU   |
| :----: | :----: | :----: |
| 0.5016 | 0.7600 | 0.5427 |

