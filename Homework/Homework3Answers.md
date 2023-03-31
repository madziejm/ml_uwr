# Machine Learning 3 Homework -- 2022-02-07

**For online delivery before 00:00 7.2.22 to your tutor**

**Points: 10**

## Problem 1 [1.5p]

Consider AdaBoost.
Prove that $\alpha_t = \frac{1}{2}\log\frac{1-\epsilon_t}{\epsilon_t}$, where $\epsilon_t$ is the error rate is the optimal coefficient for the $t$-th weak learner under the exponential loss.

---
Assume $t > 0$

From lecture we have:

$$
\begin{split}
L_t &= \frac{1}{N}\sum_{i=1}^N e^{-y^i F_t(x^i)}  = \frac{1}{N}\sum_{i=1}^N e^{-y^i \sum_{t'=1}^t\alpha_{t'} f_{t'}(x^i)} \\
&= \frac{1}{N}\sum_{i=1}^N e^{-y^i F_{t-1}(x^i)}e^{- y^i \alpha_t f_t(x^i)} \\
&= \frac{1}{N}\sum_{i=1}^N w_t^i e^{- y^i \alpha_t f_t(x^i)} \\
&= \frac{1}{N}\left[ 
    \sum_{i:y^i = f_t(x^i)} w_t^i e^{-\alpha_t} +
    \sum_{i:y^i \neq f_t(x^i)} w_t^i e^{\alpha_t}
    \right] \\
&= \frac{1}{N}\left[ 
    \sum_{i:y^i = f_t(x^i)} w_t^i e^{-\alpha_t} +
    \sum_{i:y^i \neq f_t(x^i)} w_t^i e^{\alpha_t} +
    \sum_{i:y^i \neq f_t(x^i)} \left( w_t^i e^{-\alpha_t} - w_t^i e^{-\alpha_t} \right)
    \right] \\
&= \frac{1}{N}\left[
    \sum_i w_t^i e^{-\alpha_t} + \sum_{i:y^i \neq f_t(x^i)} w_t^i \left( e^{\alpha_t} - e^{-\alpha_t}\right)
    \right] 
\end{split}
$$
We differentiate and set the derivative to zero

$$
\begin{split}
0 =
\frac{
d(\frac{1}{N}\left[ 
    \sum_{i:y^i = f_t(x^i)} w_t^i e^{-\alpha_t} +
    \sum_{i:y^i \neq f_t(x^i)} w_t^i e^{\alpha_t}
    \right] )
}{d \alpha_t} \\
&= \frac{1}{N}\left[ 
    -\sum_{i:y^i = f_t(x^i)} w_t^i e^{-\alpha_t} +
    \sum_{i:y^i \neq f_t(x^i)} w_t^i e^{\alpha_t}
    \right]
\end{split}
$$

Then

$$
\begin{split}
    \sum_{i:y^i \neq f_t(x^i)} w_t^i e^{\alpha_t} =
    \sum_{i:y^i = f_t(x^i)} w_t^i e^{-\alpha_t}
\end{split}
$$

$$
\begin{split}
     e^{\alpha_t} \sum_{i:y^i \neq f_t(x^i)} w_t^i=
     e^{-\alpha_t} \sum_{i:y^i = f_t(x^i)} w_t^i
\end{split}
$$

$$
\begin{split}
     \alpha_t  + \log \left(  \sum_{i:y^i \neq f_t(x^i)} w_t^i \right) =
     -\alpha_t + \log \left( \sum_{i:y^i = f_t(x^i)} w_t^i \right)
\end{split}
$$

$$
\begin{split}
     2 \alpha_t =
     \log \left( \sum_{i:y^i = f_t(x^i)} w_t^i \right) -   \log \left(  \sum_{i:y^i \neq f_t(x^i)} w_t^i \right)
\end{split}
$$

$$
\begin{split}
     \alpha_t = \frac{1}{2}
     \log \left(\frac{\sum_{i:y^i = f_t(x^i)} w_t^i}{\sum_{i:y^i \neq f_t(x^i)} w_t^i}  \right)
\end{split}
$$

$\epsilon_t = \frac{\sum_{i:y^i \neq f_t(x^i)} w_t^i}{\sum_{i} w_t^i}$

Thus
$$\frac{1-\epsilon_t}{\epsilon_t} = \frac{\sum_{i:y^i = f_t(x^i)} w_t^i}{\sum_{i} w_t^i}  \frac{\sum_{i} w_t^i}{\sum_{i:y^i \neq f_t(x^i)} w_t^i}$$

Now we see that the equality stated in the problem statement holds.

## Problem 2 [0.5p]

Would it make sense to apply boosting in a regression setting using linear regression as the base learner?

---

No, because sum of linear functions is still a linear function.

## Problem 3 [0.5p]

Suppose you apply AdaBoost using fully grown trees as the base learner. What would happen? What would the final ensemble look like?

---

I don't get well the intuitions behind bias and variance in ML models, but I suppose that AdaBoost works well, because it averages models' biases, so they cancel out. That's why we opt for use weak learners, which are learners with high bias and low variance. Complex models like deep decision trees have high tendency towards overfitting and high variance though, so they are not good choice in this case.

## Problem 4 [Bishop 14.7] [1p]

Consider a datasample $x$ with label distribution $p(y|x)$ (thus we assume that the sample may be ambiguous). Assume the boosted ensemble returns for this sample a score $f_x$.

The expected loss on this sample is 
$$
\mathbb{E}_{y|x}\left[e^{-yf_x}\right] = \sum_{y\in\pm 1}e^{-yf_x}p(y|x)
$$

Find the value of the model output $f_x$ which minimizes $\mathbb{E}_{y|x}\left[e^{-yf_x}\right]$ and express it in terms of $p(y=1|x)$ and $p(y=-1|x) = 1 - p(y=1|x)$.

## Problem 5 [0.5p]

Suppose you work on a binary classification problem and train 3 weak classifiers. You combine their prediction by voting. 

Can the training error rate of the voting ensemble smaller that the error rate of the individual weak predictors? Can it be larger? Show an example or prove infeasibility.

Consider a following binary categorical dataset:

---

| $X$ | $Y$ |
|:---:|:---:|
|  1  |  0  |
|  2  |  0  |
|  3  |  0  |

and following three classifiers:

| $X$ | $Y_{C1}$ |
|:---:|:--------:|
|  1  |    1     |
|  2  |    0     |
|  3  |    0     |

| $X$ | $Y_{C2}$ |
|:---:|:--------:|
|  1  |    0     |
|  2  |    1     |
|  3  |    0     |

| $X$ | $Y_{C2}$ |
|:---:|:--------:|
|  1  |    0     |
|  2  |    0     |
|  3  |    1     |

All of the above classifiers score 33% accuracy, but if they vote voted together, newly introduced assembly scores 0% accuracy.

Consider another three classifiers:

| $X$ | $Y_{C1}$ |
|:---:|:--------:|
|  1  |    1     |
|  2  |    0     |
|  3  |    0     |

| $X$ | $Y_{C2}$ |
|:---:|:--------:|
|  1  |    0     |
|  2  |    1     |
|  3  |    0     |

| $X$ | $Y_{C2}$ |
|:---:|:--------:|
|  1  |    0     |
|  2  |    0     |
|  3  |    1     |

All of the above classifiers score 66.666... % accuracy, but if they vote voted together, newly introduced assembly scores 100% accuracy.

## Problem 6 (Bishop) [1.5p]

Consider $K$-element mixtures of $D$-dimensional binary vectors. Each component of the mixture uses a different Bernoulli distribution for each dimension of the vector:

$$
\begin{split}
p(z=k) &= \pi_k \quad \text{with } 0 \leq \pi_k \leq 1 \text{ and } \sum_k\pi_k = 1\\
p(x | z=k) &= \prod_{d=1}^{D} \mu_{kd}^{x_d}(1-\mu_{kd})^{(1-x_d)}
\end{split}
$$

where $x\in\{0,1\}^D$ is a random vector. The $k$-th mixture component is parameterized by $D$ different probabilities $\mu_{kd}$ of $x_d$ being 1.

Do the following:
- Write an expression for the likelihood ($p(x;\pi,\mu)$).
- Compute the expected value of $x$.
- Compute the covariance of $x$.

## Problem 7 [1.5p]

Let $X\in \mathbb{R}^{D\times N}$ be a data matrix containing $N$ $D$-dimensional points. Furthermore assume $X$ is centered, i.e. 
$$
\sum_{n=1}^N X_{d,n} = 0 \quad \forall d.
$$

Read about the SVD matrix decomposition (https://en.wikipedia.org/wiki/Singular_value_decomposition). 

Show:
- **P7.1** [0.5p] how the singular vectors of $X$ relate to eigenvectors of $XX^T$
- **P7.2** [1p] that PCA can be interpreted as a matrix factorization method, which finds a linear projection data which retains the most information about $X$ (in the least squares sense).

## Problem 8 [2p]

Consider a symmetric real matrix A, i.e. $A\in \mathbb{R}^{D\times D}$ and $A=A^T$.

1. Show that the eigenvalues of $A$ are real. Hint: try to multiply by the Hermitean transpose of the corresponding eigenvector.

2. Show that if two eigenvalues are different, their eigenvectors are orthogonal.

---

Let $A^H$ denote Hermitean transpose.
$A = A^H$, because $X$ is real and symmetric.

Suppose that $\lambda$ is an eigenvalue of $A$ and $x \neq \vec{0}$ is a corresponding eigenvector.  
Then $Ax = \lambda x$.

---
Part 1: eigenvalues of $A$ are real
<!-- Fact: $\bar{x} x \ge 0$ (it is obvious when you multiply these two vectors)

$$Ax = \lambda x$$

$$(Ax)^H = (\lambda x)^H$$
 -->

$A = \bar{A}$, because $A$ is real.

Thus:

1. $$v^H A v =v^H (A v) =v^H (\lambda v) = \lambda \overline{v}^T v = \lambda (\overline{v} \cdot v)$$
2.   
    1.  $Av = \lambda v \implies \overline{Av} = \overline{\lambda v} \implies A \overline{v} = \overline{\lambda v}$  
        This will be used below.
    2.  $$v^H A v = \overline{v} ^T A^T v = (A \overline{v})^T v = (\overline{\lambda v})^T v = \overline{\lambda} \overline{v}^T v = \overline{\lambda} (\overline{v} \cdot v)$$

From 1. and 2.2 we see that $\lambda = \overline{\lambda}$, so $\lambda \in \mathbb{R}$

---
Part 2: if two eigenvalues are different, their eigenvectors are orthogonal

Let the two eigenvalues be $\lambda_u \ne \lambda_v$ and the eigenvectors be $u \ne v$.

Then $Auv = uA^T v$, because $A$ is symmetric.

Also

$$\lambda_u u \cdot v = \lambda_v u \cdot u$$

so

$$(\lambda_u - \lambda_v) u \cdot v = 0 $$

Considering the fact that $\lambda_u - \lambda_v$ (because they are not equal)

$$u \cdot v = 0$$

## Problem 9 (Bishop) [1p]

Recall the SVM training problem

$$
\begin{split}
\min_{w,b} & \frac{1}{2}w^T w \\
\mbox{s.t. } & y_i(w^Tx_i+b) \geq 1\qquad \textrm{for all } i.
\end{split}
$$

Show that the solution for the maximum margin hyperplane doesn't change when the $1$
on the right-hand side of the contraints is replaced by any $\gamma>0$.

