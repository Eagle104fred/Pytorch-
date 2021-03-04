# Pytorch入门
## 1.Warm-up
```python
import numpy as np

#初始化模型层数N
N,D_in,H,D_out = 64,1000,100,10

#构建输入矩阵x，输出矩阵y，参数矩阵w1，参数矩阵w2
x = np.random.randn(N, D_in)           #64*1000
y= np.random.randn(N, D_out).          #64*10
w1=np.random.randn(D_in,H).            #1000*100
w2 = np.random.randn(H,D_out).         #100*10
lr=1e-6                                #学习率learningRate

for n in range(500):
    #注意模型点积后矩阵大小
    h=x.dot(w1)                        #(64*1000)(1000*100)=64*100
    h_relu=np.maximum(h,0)             #relu 这里是一个模仿激活函数relu的操作，直接屏蔽了小于0的值
    y_pred = h_relu.dot(w2)            #(64*100)(100*10)=64*10
    loss = np.square(y_pred-y).sum()   #将矩阵元素相加方便计算y_pred和y矩阵的差值，开平方为了扩大元素值之间的差距方便更好的学习
    print(n,loss)
    
    #反向传播(求偏导的过程)
    grad_y_pred = 2*(y_pred-y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)
 
    # 更新权重
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2
```
前向传播：
-  ![](https://latex.codecogs.com/svg.latex?h=x&space;\times&space;w_1)
-  ![](https://latex.codecogs.com/svg.latex?hRelu=ReLU(h))
-  ![](https://latex.codecogs.com/svg.latex?yPred=hRelu&space;\times&space;w_2)
-  ![](https://latex.codecogs.com/svg.latex?loss=(yPred-y)^2) 
 
反向传播：
实际上就是反向求偏导。
- E.g. 计算loss对w2的偏导过程如下：
-  ![](https://latex.codecogs.com/svg.latex?\frac{\partial&space;loss}{\partial&space;w_2}=\frac{\partial&space;loss}{\partial&space;yPred}&space;\times&space;\frac{\partial&space;yPred}{\partial&space;w_2})![](https://latex.codecogs.com/svg.latex?=)![](https://latex.codecogs.com/svg.latex?2(yPred-y)hRelu)

