title: '[学习日志]2020 DIGIX全球校园AI算法精英大赛——赛道B'
author: meurice
date: 2020-07-28 18:06:11
tags:
---
## 2020.7.28
### Resnet50预训练权重文件
　　.h5文件已上传至百度网盘，链接放在此处。  
   　　[resnet50_weights_tf_dim_ordering_tf_kernels.h5](https://pan.baidu.com/s/1jTn1lI101BZfOoFys9tlOA)，提取码: pdcg  
      　　放在C://users//(yourusername)//.keras//models文件下。  
         　　另外，可以通过[该网站](https://d.serctl.com/)下载Github上的release内容。
### plt.imshow与cv2.imshow显示色差
　　使用plt.imshow和cv2.imshow对同一幅图显示时，可能会出现色差，这是由于opencv的接口为BGR，而matplotlib.pyplot接口使用的是RGB。
  ```Python
  img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
  
  plt.imshow(img)
  plt.show()
  ```
　　或通过以下方法也可实现：
  ```
  b,g,r = cv2.split(cv2.imread(img_path))
  img = cv2.merge([r,g,b])
  ```
## 2020.8.7
### 余弦相似度
　　余弦相似性通过测量两个向量的夹角的余弦值来度量它们之间的相似性。0度角的余弦值是1，而其他任何角度的余弦值都不大于1；并且其最小值是-1。从而两个向量之间的角度的余弦值确定两个向量是否大致指向相同的方向。两个向量有相同的指向时，余弦相似度的值为1；两个向量夹角为90°时，余弦相似度的值为0；两个向量指向完全相反的方向时，余弦相似度的值为-1。该结果仅与向量方向相关。余弦相似度通常用于正空间，因此给出的值为-1到1之间。  
　　![](https://wx2.sbimg.cn/2020/08/08/oJscK.png)
　　给定两个属性向量，A和B，其余弦相似性θ由点积和向量长度给出：
　　![](https://wx2.sbimg.cn/2020/08/08/oJLVT.png)
　　对于两个向量的**余弦距离**（余弦距离 = 1 - 余弦相似度）的基本计算，Python代码如下：
  ```Python
  def cosin_distance(vec_1, vec_2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vec_1, vec_2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2
    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA * normB) ** 0.5)
 ```
## 2020.8.8
### 大规模数据下使用faiss计算余弦相似度(待完善)
  ```Python
  d = 2048                           # dimension

  nb = gallery_features.shape[0]        # database size
  nq = query_features.shape[0]      # nb of queries

  xb = gallery_features.astype('float32')
  xq = query_features.astype('float32')


  nlist = 1000                      #聚类中心的个数
  k = 10      # topk搜索
  quantizer = faiss.IndexFlatL2(d)  # the other index
  index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
       # here we specify METRIC_L2, by default it performs inner-product search
  assert not index.is_trained
  index.train(xb)
  assert index.is_trained
 
  index.add(xb)                  # add may be a bit slower as well
  D, I = index.search(xq, k)     # actual search
  index.nprobe = 10              # default nprobe is 1, try a few more
  D, I = index.search(xq, k)

  ```
　　此处参考[官方样例](https://github.com/facebookresearch/faiss/wiki/Getting-started)。
## 2020.8.11
### Keras添加网络结构报错
  ```Python
  model = Sequential()
  model.add(load_model('/mnt/resnet.model').get_output_at(0))
  ```
　　*TypeError: The added layer must be an instance of class Layer.*   
　　可能是混合使用了keras.Sequential()和tf.keras.Sequential()；Keras的layer中有input和output属性，错误地使用该部分的成员函数时也可能导致该问题。  
　　修改如下：
  ```Python
  model = Sequential()
  model.add(load_model('/mnt/resnet.model').get_layer(index=0))
  ```
　　