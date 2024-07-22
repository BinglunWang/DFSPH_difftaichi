import taichi as ti
import taichi.math as tm

ti.init(arch = ti.gpu)


x = ti.field(ti.f32)
y = ti.field(ti.f32)
ti.root.dense(ti.ij, (3, 4)).place(x, y)
# is equivalent to:
x = ti.field(ti.f32, shape=(3, 4))

print(x)

print(y)