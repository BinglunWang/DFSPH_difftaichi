import taichi as ti
ti.init()

N = 16
step = 3
weight = 1e-5

x = ti.field(dtype = ti.f32, shape = (step), needs_grad = True)
y = ti.field(dtype = ti.f32, shape = (), needs_grad = True)


@ti.kernel
def func():
    for i in range(step):
        f1 = (x[i] + 1) ** 2
        f2 = x[i] ** 3
        tmp = f1 + weight * f2
        y[None] = y[None] + tmp
    # loss[None] = (y[None] - target[None]) ** 2  


for i in range(step):
    x[i] = i
    
y[None] = 0
y.grad[None] = 1


func()
func.grad()

for i in range(step):
    print(x.grad[i])