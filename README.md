# Fengyun-XEUVI
To process XEUVI data from Fengyun-3E


#0329记录
添加func_zjl_v2.py image_align0328.py
实现以下新功能：
1. 数据文件区分成几轨来处理，可以一次性处理长时间的数据
2. 下载aia数据可以选用sunpy自带fido或jsoc网站爬虫，jsoc下载比较稳定，但分辨率偏低些
3. 下载aia数据添加了尝试次数阈值
4. 每一轨选取中间时刻的XEUVI图像与aia图像对齐，并将其作为参考图像对齐其他帧
5. 程序运行过程中添加了部分文字提示
6. 将每一轨数据打包，存成fits文件（已注释）
7. 绘制图像，添加一个绘制局部区域的panel
