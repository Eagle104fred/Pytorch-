# Pytorch入门
## 1.Warm-up
```python
import numpy as np
N,D_in,H,D_out = 64,1000,100,10
x = np.random.randn(N, D_in)     #64*1000
y= np.random.randn(N, D_out).    #64*10
w1=np.random.randn(D_in,H).      #1000*100
w2 = np.random.randn(H,D_out).   #100*10
lr=1e-6

for n in range(500):
    h=x.dot(w1)                  #64*100
    h_relu=np.maximum(h,0).      #relu
    y_pred = h_relu.dot(w2).     #64*10
    loss = np.square(y_pred-y).sum()
    print(n,loss)
    
    #backprop
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


