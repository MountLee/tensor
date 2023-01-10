# library(rTensor)
# 
# a = array(c(1:27), dim = c(3,3,3))
# 
# a[1,,] = matrix(c(1:9), ncol = 3, byrow = TRUE)
# a[2,,] = matrix(c(10:18), ncol = 3, byrow = TRUE)
# a[3,,] = matrix(c(19:27), ncol = 3, byrow = TRUE)
# 
# # a[1,,] = matrix(c(1:9), ncol = 3, byrow = F)
# # a[2,,] = matrix(c(10:18), ncol = 3, byrow = F)
# # a[3,,] = matrix(c(19:27), ncol = 3, byrow = FALSE)
# 
# a
# 
# a = as.tensor(a)
# k_unfold(a, 1)
# k_unfold(a, 2)
# k_unfold(a, 3)
# 
# 
# 
# 
# #### simple case ####
# 
# a = array(c(1:8), dim = c(2,2,2))
# 
# a
# 
# a = as.tensor(a)
# k_unfold(a, 1)
# k_unfold(a, 2)
# k_unfold(a, 3)


library(rTensor)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
library(Rcpp)

library(rbenchmark)

Rcpp::sourceCpp("cpd_hpca.cpp")
Rcpp::sourceCpp("tensor_utils.cpp")

source("tensor_functions.R")
#### how to use tensor


# test(c(1,2,3))



a = matrix(runif(10000, 0, 1), ncol = 100)

benchmark(
  R       = svd(a), 
  cpp     = svd_cpp(a),
  order="relative", replications = 50)[,1:4]


a = array(c(1:8), dim = c(2,2,2))
a

check_copy(a)


a = array(c(1:8), dim = c(2,2,2))
a

b = shape(a)

a[1,,]
check(a)


### tensor times matrix

## check correctness
a = array(c(1:8), dim = c(2,2,2))
a = as.tensor(a)
b = array(rep(1, 4), dim = c(2,2))
k = 1
d = rTensor::ttm(a, b, k)@data
d

a = array(c(1:8), dim = c(2,2,2))
d = ttm_cpp(a, b, 1)
d

a = array(c(1:8), dim = c(2,2,2))
at = as.tensor(a)
b = array(c(1:4), dim = c(2,2))

k = 3
d = rTensor::ttm(at, b, k)@data
d

d = ttm_cpp(a, b, k)
d

## check speed

n1 = 20
n2 = 20
L = 10
N = n1 * n2 * L

a = array(runif(N, 0, 1), dim = c(n1, n2, L))
at = as.tensor(a)
k = 1

b = array(runif(n1 * n2, 0, 1), dim = c(n1, n2))

benchmark(
  R       = rTensor::ttm(at, b, k)@data, 
  cpp     = ttm_cpp(a, b, k),
  order="relative", replications = 100)[,1:4]




### axis permutation

a = array(c(1:8), dim = c(2,2,2))


b = permute(a, c(1,2,3))
b

b = permute(a, c(3,2,1))
b

b = array(c(1:4), dim = c(2,2))
d = reshape_cpp(b)


### k-unfold

## correctness
a = array(c(1:8), dim = c(2,2,2))
a
b = k_unfold_cpp(a, 3)


at = as.tensor(a)
k_unfold(at, 3)




b = k_unfold_cpp(a, 1)
b
at = as.tensor(a)
k_unfold(at, 1)



b = k_unfold_cpp(a, 2)
b
at = as.tensor(a)
k_unfold(at, 2)


## speed
n1 = 20
n2 = 20
L = 10
N = n1 * n2 * L

a = array(runif(N, 0, 1), dim = c(n1, n2, L))
at = as.tensor(a)

benchmark(
  R       = k_unfold(at, 3), 
  cpp     = k_unfold_cpp(a, 3),
  order="relative", replications=1000)[,1:4]






# a = array(c(1:8), dim = c(2,2,2))
# 
# a[1,,] = matrix(c(1:4), ncol = 2, byrow = TRUE)
# a[2,,] = matrix(c(5:8), ncol = 2, byrow = TRUE)
# 
# a
# 
# a = as.tensor(a)
# k_unfold(a, 1)
# k_unfold(a, 2)
# k_unfold(a, 3)


### pca test

## correctness
a = array(c(1:4), dim = c(2,2))
b = hetero_pca_test_cpp(a, 10, 20, 1e-6)



a = array(c(1:8), dim = c(2,2,2))
a
b = tensor_hetero_pca_test_cpp(a, c(10,10,10))


## speed
n1 = 20
n2 = 20
L = 10
N = n1 * n2 * L

a = array(runif(N, 0, 1), dim = c(n1, n2, L))
at = as.tensor(a)

r_hat = c(10, 10, 10)

benchmark(
  R       = Tensor_Hetero_PCA_test(at, r_hat), 
  cpp     = tensor_hetero_pca_test_cpp(a, r_hat),
  order="relative", replications=50)[,1:4]





### tensor estimation

source("cpd_MRDPG_functions.R")

frobenius <- function(A, B){
  return (sum((A - B)^2)^0.5)
}


n_1 = 50
n_2 = 50
n = n_1
d = 4
L = 2 


set.seed(0)


probability_1 = array(NA, c(n_1, n_2, L))

prob = seq(0,1,  1/(2*L))
for (layer in 1: L)
{
  p_1 =  runif(1, prob[L+layer], prob[L+layer+1])
  p_2 = runif(1, prob[layer], prob[layer+1])
  P =  matrix(p_1,n,n)
  P[1:floor(n/4), 1:floor(n/4)] = p_2
  P[(1+floor(n/4)):(2*floor(n/4)),(1+floor(n/4)):(2*floor(n/4)) ] = p_2
  P[(1+2*floor(n/4)):(3*floor(n/4)),(1+2*floor(n/4)):(3*floor(n/4)) ] = p_2
  P[(1+3*floor(n/4)):n,(1+3*floor(n/4)):n ] = p_2
  probability_1[, , layer] = P
  # probability_2[, , L - layer + 1] = P
}

set.seed(0)

A = array(0, c(n_1, n_2, L))
A = generate_tensor_probability(n_1, n_2, L, probability_1)$Y.tensor@data
At = as.tensor(A)

r_hat = rep(10, 3)
P_hat = hetero_pca_estimate_cpp(A, r_hat)
frobenius(P_hat, probability_1)


# source("cpd_MRDPG_functions.R")
# require(rbenchmark)
# library(rTensor)
# Rcpp::sourceCpp("tensor_utils.cpp")

benchmark(
  R       = hetero_pca_estimate(At, r_hat), 
  cpp     = hetero_pca_estimate_cpp(A, r_hat),
  order="relative", replications=50)[,1:4]



### operations on list of tensors
TT = 10
A_list = list()
for (i in 1:TT){
  A_list[[i]] = array(c(1:8), dim = c(2,2,2)) 
}


check_tensor_list(A_list, 0.1)

tensor_sum(A_list, 0.1)




#### uase

### cs_unfold is the same as k_unfold_cpp
a = array(c(1:8), dim = c(2,2,2))
a

at = as.tensor(a)

k = 1
b = cs_unfold(at, k)
b
k_unfold(at, k)
k_unfold_cpp(a, k)

b = t(k_unfold_cpp(a, 1))
d = fold_cpp(b, 2, 2, 2)
d
a


array(b, dim = c(2,2,2))

#### fold_cpp is slower than R, though both times are negligible
n = 1000
r = 10
a = array(c(1:n), dim = rep(r,3))
b = t(k_unfold_cpp(a, 1))
benchmark(
  R       = array(b, dim = c(r, 3)), 
  cpp     = fold_cpp(b, r, r, r),
  order="relative", replications=5000)[,1:4]

# Rcpp::sourceCpp("tensor_utils.cpp")


vec2mat(c(1:4), 2, 2)
vec2mat_manual(c(1:4), 2, 2)
matrix(c(1:4), 2, 2)

n = 10000
r = 100
c = 100
benchmark(
  R       = matrix(c(1:n), r, c), 
  cpp     = vec2mat(c(1:n), r, c),
  cpp_manual     = vec2mat_manual(c(1:n), r, c),
  order="relative", replications=500)[,1:4]



n1 = 50
n2 = 50
n3 = 4
n = n1 * n2 * n3
a = array(runif(n, 0, 1), dim = c(n1, n2, n3))
at = as.tensor(a)

b1 = uase_cpp(a, 3)
b2 = uase(at, 3)

sum((b1 - b2)^2)

benchmark(
  R       = uase(at, 3), 
  cpp     = uase_cpp(a, 3),
  order="relative", replications=500)[,1:4]

