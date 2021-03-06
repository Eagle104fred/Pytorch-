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
对比之前的操作，会发现Pytorch大大降低了手工操作的负担，只需要在设定的时候增加requires_grad=True，在最后对权重进行归零即可。
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
        
        #梯度归零
        w1.grad.zero_()
        w2.grad.zero_()

```
## 4.Define my autograd function
在底层，每次原始的autograd操作都是对Tensor的两个方法的操作。

- forward方法用于计算输入Tensor
- backward方法获取输出的梯度，并且计算输入Tensors相对于该相同标量值的梯度
在Pytorch中，可以容易定义自己的autograd操作，通过定义子类torch.autograd.Function来实现forward和backward函数，然后就可以通过构建实例并进行调用来使用新的autograd运算符。传递包含输入数据的Variables。

```python
import torch
import torch.nn

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x):
        ctx.save_for_backward(x)
        return x.clamp(min = 0)
    
    @staticmethod
    def backward(ctx,grad_output):
        x, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x<0]=0
        return grad_x

device = torch.device('cuda')
N,D_in,H,D_out = 64,1000,100,10

x = torch.randn(N,D_in,requires_grad=True)
y = torch.randn(N,D_out,requires_grad=True)


device = torch.device('cuda')
N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

w1 = torch.randn(D_in, H, device=device, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    #调用自己的自动求导函数实现前向反向传播
    y_pred = MyReLU.apply(x.mm(w1)).mm(w2)

    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    loss.backward()

    with torch.no_grad():
       
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()
```




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
- 总结：梯度累加就是，每次获取1个batch的数据，计算1次梯度，梯度不清空，不断累加，累加一定次数后，根据累加的梯度更新网络参数，然后清空梯度，进行下一次循环。一定条件下，batchsize越大训练效果越好，梯度累加则实现了batchsize的变相扩大，如果accumulation_steps为8，则batchsize ‘变相’ 扩大了8倍，是我们这种乞丐实验室解决显存受限的一个不错的trick，使用时需要注意，学习率也要适当放大。
[梯度累加](https://blog.csdn.net/weixin_45997273/article/details/106720446)
### 5-1.Gradient Accumulate
```python
import torch.nn as nn

device = torch.device('cuda');
N,D_in,H,D_out = 64,1000,100,10;

x = torch.randn(N,D_in,device=device);
y = torch.randn(N,D_out,device=device);

model = nn.Sequential(
    nn.Linear(D_in,H),
    nn.ReLU(),
    nn.Linear(H,D_out),
).to(device)
lr = 0.01

accumulation_steps = 8;
optimizer = torch.optim.Adam(model.parameters(),lr)

loss_fn = nn.MSELoss(reduction='sum')

for n in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred,y)
    print(n,loss.item())
    
    loss = loss/accumulation_steps;   
    # 2.2 back propagation
    loss.backward()
    # 3. update parameters of net
    if((n+1)%accumulation_steps)==0:
        # optimizer the net
        optimizer.step()        # update parameters of net
        optimizer.zero_grad()   # reset gradient
```

## 6.Optim
运用优化功能，自动计算模型的参数
```python
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

## 7.Custom nn Modules
```python
import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    #实例化两个参数矩阵w1w2
    def __init__(self,D_in,H,D_out):
        super(TwoLayerNet,self).__init__();
        self.linear1 = nn.Linear(D_in,H);
        self.linear2 = nn.Linear(H,D_out);
        
    def forward(self,x):
        '''
        在Forward函数中，前向传播过程，我们接受输入数据的张量，并且必须返回
        输出数据的Tensor。我们可以使用构造函数中定义的Modules
        以及Tensors上的任意（可微分）操作。
        '''
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred
N,D_in,H,D_out = 64,1000,100,10;

x = torch.randn(N,D_in);
y = torch.randn(N,D_out);

model = TwoLayerNet(D_in,H,D_out);

loss_fn = nn.MSELoss(reduction='sum');

lr = 1e-4;
optimizer = torch.optim.SGD(model.parameters(),lr);
for n in range(500):
    y_pred = model(x);
    
    loss = loss_fn(y_pred,y);
    print(n,loss.item());
    
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
```

## 8.ControlFlow-WightSharing
```python
import random
import torch
import torch.nn as nn

class DynamicNet(nn.Module):
    def __init__(self,D_in,H,D_out):
        super(DynamicNet,self).__init__();
        self.input_linear = nn.Linear(D_in,H);
        self.middle_linear = nn.Linear(H,H);
        self.output_linear = nn.Linear(H,D_out);
        
    def forward(self,x):
    '''
        对于模型的前向传递，我们随机选择0、1、2或3，然后多次重复使用middle_linear
        模块来计算隐藏层表示。

        由于每个前向传播都会构建一个动态计算图，因此在定义模型的前向传播时，
        我们可以使用常规的Python控制流操作符，如循环或条件语句。

        在这里我们还看到，在定义计算图时，多次重用同一个模块是完全可行的，
        这是Lua Torch对于每个模块只能使用一次的一大改进
    '''
        h_relu = self.input_linear(x).clamp(min = 0);
        for _ in range(random.randint(0,3)):
            h_relu = self.middle_linear(h_relu).clamp(min = 0);
        y_pred = self.output_linear(h_relu);
        return y_pred;
    
    
    
N,D_in,H,D_out = 64,1000,100,10;

x = torch.randn(N,D_in);
y = torch.randn(N,D_out);

model = DynamicNet(D_in,H,D_out);

loss_fn = torch.nn.MSELoss(reduction='sum');

#使用普通的随机梯度下降训练这种奇怪的模型很困难，因此我们使用momentum(动量)
optimizer = torch.optim.SGD(model.parameters(),lr=1e-4,momentum=0.9);

for n in range(500):
    y_pred = model(x);
    loss = loss_fn(y_pred,y);
    print(n,loss.item());
    
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
```
### momentum
“冲量”这个概念源自于物理中的力学，表示力对时间的积累效应。在普通的情况下x的更新 在加上冲量后就是在普通的情况下加上上次更新的x的与mom[0,1]的乘积
- ![](https://img-blog.csdnimg.cn/2020051516503830.png)
- 当本次梯度下降- dx * lr的方向与上次更新量v的方向相同时，上次的更新量能够对本次的搜索起到一个正向加速的作用。
- 当本次梯度下降- dx * lr的方向与上次更新量v的方向相反时，上次的更新量能够对本次的搜索起到一个减速的作用。
