# Machine Learning Homework 2 2021-12-05

**Points: 10 + 3bp bonus point**


## Problem 1 [1p]

Consider a binary classification problem with discrete features, which we encode using the [one-hot](https://en.wikipedia.org/wiki/One-hot) encoding. E.g. suppose we have 2 features, $x_1\in{a,b}$ and $x_2\in{s,t,u}$. Then the encoded data is a real vector $x_e\in\mathbb{R}^5$:


| $x_1$, $x_2$ | $x_e$ |
|:------------:|:-----:|
|     a,s      | 10100 |
|     b,s      | 01100 |
|     ...      |  ...  |
|     b,u      | 01001 |


Suppose you train a logistic regression classifier and a naive Bayes one.

Show that the decision boundary of the two classifiers has the same form, i.e. that you could e.g. use a linear regression implementation to simulate the behavior of a naive Bayes model .

## Problem 2 (Weighted least squares) [1p + 1bp]

*Consider a least squares problem in which we apply a (known) weigth $w^{(i)}$ to each example:*

$$
\min_\Theta \frac{1}{2}\sum_i w^{(i)}(x^{(i)}\Theta - y^{(i)})^2.
$$

*Write down the closed-form formula for predicting a single value at $x$.*

*The weighted least squares are used in the Local "[LOESS](https://en.wikipedia.org/wiki/Localregression)" regression: we find the set of closest neighbors to a query point, weight them based on the distance to the query, then predict the answer according to the weighted least squares.*

*Describe the LOESS method for the bonus point.*

In vanilla linear regression we had:

* $J(\Theta) = \frac{1}{N}\sum_{i=1}^N (y^{(i)} - x^{(i)}\Theta)^2 = \frac{1}{N}(X\Theta - Y)^T(X\Theta - Y)$
* $\frac{\partial J(\Theta)}{\partial\Theta} = \frac{1}{N}(2 X^T X \Theta -2 X^T Y)$
* Solving for zero went like this:  
  1. $X^T X \Theta = X^T Y$  
  2. $\Theta = (X^T X) ^{-1} X^T$

After adding weights we have:

* $J(\Theta) = \frac{1}{N}\sum_{i=1}^N (y^{(i)} - x^{(i)}\Theta)^2 = \frac{1}{N} \text{diag} (w)(X\Theta - Y)^T(X\Theta - Y)$
* $\frac{\partial J(\Theta)}{\partial\Theta} = \frac{1}{N} \text{diag}(w) (2 X^T X \Theta -2 X^T Y)$
* Solving for zero goes like this:  
  1. $\text{diag}(w) X^T X \Theta = \text{diag}(w) X^T Y$  
  2. $\Theta = (X^T X) ^{-1}  \text{diag}(w^{-1}) X^T$



## Problem 3 (McKay 4.1) [1p]

You are given a set of 12 balls in which:
- 11 balls are equal
- 1 ball is different (either heavier or lighter).

You have a two-pan balance. How many weightings you must use to detect toe odd ball?

*Hint:* A weighting can be seen as a random event. You can design them to maximize carry the most information, i.e. to maximize the entropy of their outcome.

## Problem 4 (Murphy, 2.17) [1p]

*Expected value of the minimum.*

- *Let $X, Y$ be sampled uniformily on the interval $[0,1]$. What is the expected value of $\min(X,Y)$?*
- *Let $X,Y$ be independent and exponentialy distributed with* $X \sim \exp(\lambda_x), Y \sim \exp(\lambda_y)$. *What is the expected value of $\min(X,Y)$?*

1. $X, Y$ - iid. from $U_{[a, b]}$ for $a=0, b=1$,   
   PDF: $f(x) = 1 / (b - a)$ for $x \in [a, b]$ and $0$ otherwise   
   $f_X(x) = 1$ for $x \in [a, b]$ and $0$ otherwise (obviously the same for $f_Y$)   
   Introduce a new random variable $Z = \min(X, Y)$
   $\mathbb{E}Z = ? = \int_{\mathbb{R}} z f(z) dz = \int_0^1 z f(z) dz$
   $f_Z(z) = ?$   
   CDF of Z: $F(z) = P(Z \le z) = 1 - P(Z \gt z) = 1 - P(\min(X, Y) \gt z) = 1 - P(X \gt z) P(Y \gt z) = 1 - (\frac{b - z}{b - a})^2 = 1 - (1 - z)^2 = 2z - z^2$  for $x \in [a, b]$ ($0$ for $z \lt a$; $1$ for $z \gt b$)    
   PDF of Z: $f(z) = \frac{\partial{F(z)}}{\partial{z}} = 2 - 2z$   
   $\mathbb{E}Z = \int_0^1 z(2 - 2z) dz = (z^2 - \frac{2}{3}z^3) \Big|_0^1 = \frac{1}{3}$
2. $\mathbb{E}Z = ? = \int_{\mathbb{R}} z f(z) dz$   
   Mind: $0 \le X, Y, Z$  
   $f_X(x) = \lambda_x \exp \{ - \lambda_x x \}$ ($f_Y$ alike)   
   $F_X(x) = 1 - \exp \{ - \lambda_x x \}$ ($f_Y$ alike)   
   $F(z) = P(Z \le z) = 1 - P(Z \gt z) = 1 - P(\min(X, Y) \gt z) = \\ =1 - P(X \gt z) P(Y \gt z) = 1 - (1 - F_X(z))(1 - F_Y(z)) = \\= 1 - \exp(-\lambda_x z) \exp(-\lambda_y z) =\\
   = 1 - \exp(-(\lambda_x + \lambda_y) z)$   
   This is CDF for exponential distribution, thus $Z \sim \exp(\lambda_y + \lambda_y)$
   Let's exploit exponential distribution expected value wzórek then: $\mathbb{E}Z = \frac {1}{\lambda_x + \lambda_y}$
   


## Problem 5 [1p]

*Find the gradient with respect to $x$ of $- \log(S(\mathbf{x})_j)$, where $S$ is softmax function (https://en.wikipedia.org/wiki/Softmaxfunction) and we want derive the derivative over the $j$-th output of the Softmax.*

Let's follow Wikipedia notation

$j$-th output of Softmax: $\sigma(z)_j = \frac{\exp(z_j)}{\sum_{1 \le i \le K} \exp(z_i)}$ for $1 \le j \le K$ and $z = (z_1, \cdots, z_K) \in \mathbb{R}^K$

---
What we want is $j$-th output of Softmax partial derivative with respect to $i$-th input (**in fact not, see below**):
$\frac{\partial{ \sigma(z)_j}}{\partial{z_i}} = \frac{\partial}{\partial{z_i}} \frac{\exp(z_j)}{\sum_{1 \le i \le K} \exp(z_i)} = \cdots$

Recap:  
$\frac{\partial}{{\partial x}}\left( {\frac{{f\left( x \right)}}{{g\left( x \right)}}} \right) = \frac{{\frac{d}{{dx}}f\left( x \right)g\left( x \right) - f\left( x \right)\frac{d}{{dx}}g\left( x \right)}}{{g^2 \left( x \right)}}$

Here we have
$f(x) = \exp(z_j)$
$\frac{\partial f(x)}{{\partial z_i}} = \exp(z_j)$ if $i =j$ and $0$ otherwise
$g(x) = \sum_{1 \le i \le K} \exp(z_i)$
$\frac{\partial g(x)}{{\partial z_i}} = \exp(z_i)$

Let $N := \sum_{1 \le i \le K} \exp(z_i)$

Follow-up:
We see we have two cases

1. $i=j$ case  
   $\cdots = \frac{\exp(z_j) N - \exp(z_j)\exp(z_i)}{N^2} = \frac{\exp(z_j)}{N} - \frac{\exp(z_j)}{N} \frac{\exp(z_i)}{N} = \sigma(z)_j (1 - \sigma(z)_i)$  
   
2. $i \neq j$ case  
   $\cdots = \frac{0 - \exp(z_j) \exp(z_i)}{N^2} = -\sigma(z)_j \sigma(z)_i$  

---

But we were asked to find derivative of log in the first place:

Recap:  
By chain rule $\frac{d}{dx} \log(f(x)) = \frac{\frac{df(x)}{dx}}{f(x)}$

$\log(\sigma(z)_j) = \log \exp(z_j) - \log \sum_{1 \le k \le K} \exp(z_k) = z_j - \log \sum_{1 \le k \le K} \exp(z_k)$

1. $i=j$ case  
   $\frac{\partial \log(\sigma(z)_j)}{\partial z_i} = 1 - \frac{z_i \exp(z_i)}{\sum_{1 \le k \le K} \exp(z_k)} = 1 - z_i\sigma(z)_i$
2. $i \neq j$ case  
   $\frac{\partial \log(\sigma(z)_j)}{\partial z_i} = - \frac{z_i \exp(z_i)}{\sum_{1 \le k \le K} \exp(z_k)} = - z_i \sigma(z)_i$
<!--    $\log(\frac{\partial{ \sigma(z)_j}}{\partial{z_i}}) = \log \Big( \frac{\exp(z_j) \exp(z_i)}{N^2} \Big) = z_j - z_i + 2 N = z_j - z_i + 2 \sum_{1 \le i \le K} \exp(z_i)$ -->

Thus $\nabla \sigma(z)_j = \sigma(z) + \begin{cases}
0 & i \neq j \\
1 & i = j \\
\end{cases}$

(rhs is $K$-element vector indexed by $i$)

Aside: $z_i \sigma(z)_i$ looks kinda like expected value.

---

<!-- By chain rule $\nabla \log(f(x)) = \frac{ \nabla f(x)}{f(x)}$

$\log(\sigma(z)_j) = \log \exp(z_j) - \log \sum_{1 \le i \le K} \exp(z_i) = z_j - \log \sum_{1 \le i \le K} \exp(z_i)$

1. $i=j$ case  
   $\frac{\partial \log(\sigma(z)_j)}{\partial z_i} = - \frac{\sum_{1 \le i \le K} z_i \exp(z_i)}{\sum_{1 \le i \le K} \exp(z_i)} = z_j - \sum_{1 \le i \le K} \exp(z_i)\sigma(z)_j$
3. $i \neq j$ case  
   $\frac{\partial \log(\sigma(z)_j)}{\partial z_i} = z_j - \frac{\sum_{1 \le i \le K} z_i \exp(z_i)}{\sum_{1 \le i \le K} \exp(z_i)} = z_j - \sum_{1 \le i \le K} \exp(z_i)\sigma(z)_j$
<!--    $\log(\frac{\partial{ \sigma(z)_j}}{\partial{z_i}}) = \log \Big( \frac{\exp(z_j) \exp(z_i)}{N^2} \Big) = z_j - z_i + 2 N = z_j - z_i + 2 \sum_{1 \le i \le K} \exp(z_i)$ --> -->

## Problem 6 [1p]

Consider regresion problem, with $M$ predictors $h_m(x)$ trained to aproximate a target $y$. Define the error to be $\epsilon_m(x) = h_m(x) - y$.

Suppose you train $M$ independent classifiers with average least squares error
$$
E_{AV} = \frac{1}{M}\sum_{m=1}^M \mathbb{E}_{x}[\epsilon_m(x)^2] = 1/M \sum V_i.
$$

Further assume that the errors have zero mean and are uncorrelated:
$$
\mathbb{E}_{x}[\epsilon_m(x)] = 0\qquad\text{ and }\qquad\mathbb{E}_{x}[\epsilon_m(x)\epsilon_l(x)] = 0\text{ for } m \neq l
$$

Let the mean predictor be
$$
h_M(x) = \frac{1}{M}\sum_{m=1}^Mh_m(x).
$$

What is the average least squares error of $h_M(x)$?

$V_i = E_i ^ 2$

$\epsilon_M(x)^2 = (1/ M \sum_{1 \le i \le M} \epsilon_i(x) \Big)^2 = 1/M \sum V_i$


## Problem 7: Numerical stability of SoftMax [1p]

*Many classifiers, such as Naive Bayes, multiply probabilities. To prevent loss of precision due to numerical underflow, we often prefer to add log-probabilities, rather than to multiply probabilities. However, we sometimes also need to add probabilities. This happens e.g. during normalization:*
- *in Naive Bayes, to get final probabilities we need to compute a sum-exp of the log-scores.*
- *in Softmax, we again compute a sum-exp of the scores.*

*Going back to logarithms requires us to compute a log-sum-exp operation, or in other words using exponentiation, then addition, and finally re-application of the logarithm.*

*Explain when the log-sum-exp operation may fail numerically, and how this failure can be prevented. Show how this fix can also be applied to the SoftMax function.*

Softmax takes as input a vector of some $K$ real numbers, and normalizes it into a probability distribution consisting of $K$ probabilities proportional to the exponentials of the input numbers.

Problem: exponentiation in the softmax function makes it possible to easily explode this number, even for normal-sized inputs.

Example: let's say we want to normalize $[1000, 1000, 1000]$. This should obviously yield $[1/3, 1/3, 1/3]$, but for each term we need to calculate $\frac{\exp(1000)}{3 \exp(1000)}$ (see Softmax formula above) which is obviously numerically instable and will overflow.

Solution: Let's rewrite Softmax as follows:
$\sigma(z)_j = \frac{\exp(z_j)}{\sum_{1 \le i \le K} \exp (z_i)} = \frac{\exp(z_j)}{\sum_{1 \le i \le K} \exp (z_i)} \frac{\exp(C)}{\exp(C)} = \frac{\exp(z_j + C)}{\sum_{1 \le i \le K} \exp (z_i + C)}$

A good choice, assuming that all the inputs are not far from each other, is $C = - \max_{1 \le i \le K} z_i$. Then large exponents will be shifted towards zero.

Next example: given $\log x_1, \cdots, \log x_n$ we want to calculate $\log \sum_{1 \le i \le n} x_i$. Then we can calculate it as follows $\log(\exp(C) \sum_{1 \le i \le n} x_i) = C + \log \sum_{1 \le i \le n} \exp(x_i - C)$ for $C = \max_{1 \le i \le K} x_i$

## Problem 8.1 [1p]

We have $X_1, X_2, X_3$ random variables such that correlation of each two of them is equal to $\rho$. Find the smallest value of $\rho$. 

---

Answer for this one goes below (8.2).

## Problem 8.2 [2bp]

We have $X_1, X_2, X_3, ..., X_n$ random variables such that correlation of each two of them is equal to $\rho$. Find the smallest value of $\rho$.

---

$\rho_{X, Y} = \operatorname{corr}(X,Y) = \frac{\operatorname{cov}(X,Y)}{\sigma_X \sigma_Y} = \frac {\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]]}{\sigma_X\sigma_Y} \in [-1. 1]$

This is called Pearson product-moment correlation coefficients.

Spoiler: lowest bound for $\rho$ is in this case $\frac{1}{2}$

Question statement yields the following correlation matrix:
$\Sigma = \begin{matrix}
      1 & \dots & \rho & \dots & \rho \cr
   \vdots & \ddots & \vdots & \ddots & \vdots \cr
   \rho & \dots & 1 &  \dots & \rho \cr
   \vdots & \ddots & \vdots & \ddots & \vdots \cr
   \rho & \dots & \rho & \dots & 1 \cr
\end{matrix}$

$\Sigma = \rho \mathbb{1}_{n \times n} + (1 - \rho) I_{n \times n}$  
($1_{n \times n}$ is matrix of ones)


1. This matrix must be semi-definite (ie. with not negative eigenvalues)
   https://stats.stackexchange.com/questions/69114/why-does-correlation-matrix-need-to-be-positive-semi-definite-and-what-does-it-m
2. Matrix $M$ is positive semi-definite if for every non-zero column vector $z$, $0 \le z^T M z$  
3. Correlation matrix is the same as the **covariance** matrix of standarized random variables $X_i / \sigma(X_I)$
4. For **covariance** matrix $\Sigma$ we have  
    1.  $c^T \Sigma = \text{cov}(c^TX, X)$
    2.  $d^T \Sigma c = \text{cov}(d^TX, c^TX)$
    3.  Thus $c^T \Sigma c = \text{cov}(c^TX, c^TX)$ ((co)variance of linear combination of random variables $X$)


Aside: 2 types of corr matrices; matrices of population correlations and matrices sample correlations

## Problem 9 [2p]

Consider a dataset $X = (x_1,...,x_n),y = (y_1,...,y_n)$ where $x_i \in \mathbb{R}^d$ and $y_i \in \mathbb{R}$. We assume that the data generated was according to the following probabilistic model: $y_i = \Theta \cdot x_i + \epsilon_i$ for each $i$, where $\epsilon_i$ is IID error term distributed according to a zero-mean Laplace distribution, and $\Theta$ is some unknown but fixed vector. Laplace distribution has PDF of form $x \mapsto \frac{1}{2b}e^{-\frac{|x|}{b}}$. Show that the $\bar\Theta$ found by minimizing the Mean Absolute Error value of $\frac{1}{n}\sum_{i=1}^{n}|y_i - \Theta x_i|$ is in fact the Maximum Likelihood Estimator of the model. 

--------

Well.  
So first we assumed that the errors follow Laplace distribution.  
We observe the errors that are added to $\theta X$ for some  hidden $\theta$.  
And we want to discover the $\theta$.  
As we know the distribution of errors we can compute likelihood of observing a given theta if we assume fixed error distribution parameters. And then find the theta using MLE approach.


PDF: $f(\epsilon | u, b) = \frac{1}{2b} \exp \big\{ - \frac{|x - \mu|}{b} \big\}$
Likelihood function: $L(0, b, \epsilon_1, \cdots, \epsilon_n) = \prod_{i \le n} \frac{1}{2b} \exp \big\{ - \frac{|\epsilon_i|}{b} \big\}$
Log-likelihood function: $LL(\mu, b, \epsilon_1, \cdots, \epsilon_n) = - \frac{1}{2b} \sum_{1 \le i \le n} |\epsilon_i - \mu| =\\
-\frac{1}{2b} \sum_{1 \le i \le n}  |y - \Theta x|$

That's it

Aside: pochodna is as follows, but nobody cares

$\frac{d LL}{d\theta} = \frac{d}{d\theta} \frac{1}{b} \sum_{1 \le i \le n}  |y - \Theta x| =  - \frac{1}{b} \sum_{1 \le i \le n} \text{sgn}(x_i - \mu)$

