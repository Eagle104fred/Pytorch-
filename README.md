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
```python
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
## 4.Define my autograd function

## 5.nn
把隐藏层的参数矩阵封装成model,这样可以直接调用loss函数进行运算
```python
import torch
import torch.nn as nn

device = torch.device('cuda')
N,D_in,D_out,H = 64,1000,10,100
x = torch.randn(N,D_in,device=device)
y = torch.randn(N,D_out,device=device)

#把隐藏层的参数矩阵封装成model
model = nn.Sequential(
        nn.Linear(D_in,H),
        nn.ReLU(),
        nn.Linear(H,D_out),
).to(device)

#申请nn内部的loss函数
loss_fn = nn.MSELoss(reduction='sum')
lr = 1e-4
for n in range(500):
    y_pred=model(x)
    loss = loss_fn(y_pred,y)
    print(n,loss.item())
    
    #反向传播前先把梯度归零
    model.zero_grad()
    
    loss.backward()
    
    with torch.no_grad():
        for param in model.parameters():
            param -=lr*param.grad```

```
那么问题就来了为什么要警醒梯度归零呢？
    - 由于pytorch的动态计算图，当我们使用loss.backward()和opimizer.step()进行梯度下降更新参数的时候，梯度并不会自动清零。并且这两个操作是独立操作。
    - backward()：反向传播求解梯度。
    - step()：更新权重参数。
- 基于以上几点，正好说明了pytorch的一个特点是每一步都是独立功能的操作，因此也就有需要梯度清零的说法，如若不显示的进行optimizer.zero_grad()这一步操作，backward()的时候就会累加梯度，也就有了梯度累加这种trick。
[梯度累加](https://blog.csdn.net/weixin_45997273/article/details/106720446)
## 6.Optim
运用优化功能，自动计算模型的参数
···python
import torch
import torch.nn as nn

device = torch.device('cuda')

N,D_in,H,D_out = 64,1000,100,10

x=torch.randn(N,D_in,device=device)
y=torch.randn(N,D_out,device=device)

model = nn.Sequential(
    nn.Linear(D_in,H),
    nn.ReLU(),
    nn.Linear(H,D_out),
).to(device)
lr = 1e-4
optimizer = torch.optim.Adam(model.parameters(),lr)

loss_fn = nn.MSELoss(reduction='sum')

for n in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred,y)
    print(n,loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    
    #自动计算参数
    optimizer.step()#根据梯度更新网络参数简单的说就是进来一个batch的数据，计算一次梯度，更新一次网络。
···
