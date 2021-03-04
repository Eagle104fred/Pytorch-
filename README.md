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

## 3.Autograd
```
import torch
device = torch.device('cuda')

N,D_in,H,D_out = 64,1000,100,10

x=torch.randn(N,D_in,device=device)
y=torch.randn(N,D_out,device=device)
w1 = torch.randn(D_in,H,device=device,requires_grad=True)
w2 = torch.randn(H,D_out,device=device,requires_grad=True)
lr=1e-6

for n in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    
    loss = (y_pred-y).pow(2).sum()
    print(n,loss.item())
    
    loss.backward()#反向传播，自动求导
    
    with torch.no_grad():
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad
        
        # Manually zero the gradients after running the backward pass
        w1.grad.zero_()
        w2.grad.zero_()

```
