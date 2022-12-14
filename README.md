# Automatic derivation 
This project is based on Numpy

refer to https://github.com/tonggege001/MatrixFlow
# How to run
``` powershell
    git clone git@github.com:suololololo/MatrixGrad.git
    python test.py
```

# To do 
Write the derivation process 

# Basic Concepts
## 1.矩阵求导布局

标量y对标量x的求导, 可以表示为 $\partial y \over \partial x$。m维向量*y* 对标量x的求导，表示为 $\partial \mathbf y \over \partial x$。结果是个N维度向量，这个N维度向量中的每个元素都是y的每一维对x的导数。

可以注意到，在上述向量对标量的求导中，我们并没有讲明求导结果 $\partial \mathbf y \over \partial x$是 $N \times 1$ 还是 $1 \times N$,即是列向量还是行向量。这跟求导布局的概念有关。

上述问题的答案是都可以，即求导结果按照列排序还是行排序都是允许的。但是在机器学习算法中，行列排序会影响算法结果，因此对布局进行了定义。

<ul>
<li>分子布局: 求导结果按照分子的维度</li>
<li>分母布局: 求导结果按照分母的维度</li>
</ul>

example
<ol>
<li>
向量y 对标量x求导
其中y是列向量，求导结果如果按照分子布局，也将是列向量。如果按照分母布局，则是行向量。 
</li>
<li>
矩阵Y 对标量x求导
其中Y是m *n 矩阵，求导结果如果按照分子布局，则是m*n 矩阵。如果按照分母布局，则是n*m矩阵
</li>
<li>
....
</li>
</ol>
可以枚举出所有求导情况，可知分子布局和分母布局的结果相差一个转置。

同时布局规范具有一定的约定：
<ul>
<li>矩阵或向量对标量求导，以分子布局为准。</li>
<li>标量对矩阵或向量求导，以分母布局为准。</li>
<li>向量对向量求导，以分子布局的雅可比矩阵为主</li>
</ul>

## 2.矩阵求导定义
$\frac{\partial A}{\partial B}$定义为A的中的每个元素对B中的每个元素求导

$y = a^TXb$,y为标量，a为m维度向量，X为$m \times n$矩阵，b为n维向量，求解 $\partial y \over \partial X$。
$\partial y \over \partial X$ 按照分母布局，即求解结果为 $m \times n$矩阵。

根据定义可得
 $ \frac{\partial y}{\partial X} |_{ij} = \frac {\partial y} {\partial X_{ij}} = \frac {\partial a^TXb} {\partial X_{ij}} = \frac {\partial a_iX_{ij}b_j} {\partial X_{ij}} = a_ib_j $ 
因此，矩阵每个位置元素可根据上述式子求出
$$
\frac {\partial y} {\partial X} = \big(a_ib_1, a_1b_2, ... \big) = a^Tb
$$

上述做法将标量对矩阵的求导拆分成标量对标量的求导的组合，这就是利用定义求解，但当表达式复杂时，无法拆解成标量对标量的求导，该方法就不太适用。而且根据定义求解破坏了矩阵求导的整体性。

### 微分法
在多元函数中，导数与微分的关系根据全微分公式记为
$$
df = \sum_{i=1}^n \frac{\partial f}{\partial x_i} dx_i
$$
其中

$$
\sum_{i=1}^n \frac{\partial f}{\partial x_i} dx_i
$$
可以记为矩阵

$(\frac{\partial f} {\partial x_1}, \frac{\partial f} {\partial x_2}, ..., \frac{\partial f} {\partial x_n})$与矩阵
$
(dx_1, dx_2, ..., dx_n)^T
$的内积
因此，
$$
\sum_{i=1}^n \frac{\partial f}{\partial x_i} dx_i = \frac {\partial f}{\partial x}^T d\mathbf X
$$

$
\frac {\partial f}{\partial x} (n \times 1)是梯度向量，
$ 
$
d\mathbf X (n \times 1)是微分向量
$

把多元微分中的梯度与微分的关系扩展到矩阵，可以得到
$$
df = \sum_{i=1}^{m}\sum_{j=1}^n \frac{\partial f}{\partial X_{ij}} dX_{ij} = tr(\frac{\partial f}{\partial X}^T dX) \tag 1
$$
第二个等号的中的迹运算，具有以下性质，对于具有相同尺寸的矩阵A,B
$$
tr(A^TB) = \sum_{i=1}^{m}\sum_{j=1}^nA_{ij}B_{ij} \tag 2
$$
由式子2得到矩阵A和B的内积, 为 $tr(A^TB)$

其中A为$m \times n$, B为 $m \times n$
由式子1得全微分$df$是导数$\frac{\partial f}{\partial X}(m \times n)$和 $dX(m \times n)$的内积。
因此，矩阵微分和向量微分可以统一表示为
$$
df=tr(\frac{\partial f}{\partial X}^T dX) \tag 3
$$

$$
df=\frac{\partial f}{\partial x}^T d\mathbf x
$$
