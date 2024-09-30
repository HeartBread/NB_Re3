# NB_Re3
NDVI guided Bi-directional Recurrent Reconstruction model for multispectral Reflectance time series

输入数据形状: N表示pixel个数，C表示波段数，T表示时间长度
<br> 反射率：[N, C, T]
<br> 云掩膜：[N, 1, T]
<br> NDVI：[N, 1, T]
<br> 时序间隔：[N, 1, T]

