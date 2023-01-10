// [[Rcpp::depends("RcppArmadillo")]]
#include <RcppArmadillo.h>
#include <Rcpp.h>


using namespace Rcpp;
using namespace arma;




// [[Rcpp::export]]
Mat<double> k_unfold_cpp(Cube<double>& a, const uword k){
  const uword n1 = a.n_rows;
  const uword n2 = a.n_cols;
  const uword n3 = a.n_slices;
  
  Mat<double> res;
  Mat<double> sub;
  
  switch (k){
  case 1:{
    uword nm = n2 * n3;
    res.zeros(nm, n1);
    
    for (uword i = 0; i < n1; i++){
      sub = a.row(i);
      res.col(i) = vectorise(sub, 0);
    }
    // res = reshape(res, n1, nm);
  }
    break;
  case 2:{
    uword nm = n1 * n3;
    res.zeros(nm, n2);
    
    for (uword i = 0; i < n2; i++){
      sub = a.col(i);
      res.col(i) = vectorise(sub, 0);
    }
    // res = reshape(res, n2, nm);
  }
    break;
  case 3:{
    uword nm = n1 * n2;
    res.zeros(nm, n3);
    
    for (uword i = 0; i < n3; i++){
      sub = a.slice(i);
      res.col(i) = vectorise(sub, 0);
    }
    // res = reshape(res, n3, nm);
  }
    break;
  }
  return res;
}




// [[Rcpp::export]]
mat hetero_pca_test_cpp(mat& Y, uword r, uword tmax, double vartol){
  
  mat N_t = mat(Y);
  
  uword dim = std::min(N_t.n_rows, N_t.n_cols);
  
  if (r > dim){
    r = dim;
  }
  for (uword i = 0; i < dim; i ++){
    N_t(i, i) = 0.0;
  }
  
  mat U_t = mat(N_t.n_rows, r);
  mat V_t = mat(N_t.n_cols, r);
  mat S_t = mat(r, r);
  mat tilde_N_test_t = mat(N_t.n_rows, N_t.n_cols);
  uword t = 1;
  double approx = -1.0;
  
  mat U;
  vec s;
  mat V;
  
  while(t < tmax){ 
    // Stop criterion: convergence or maximum number of iteration reached
    svd(U, s, V, N_t);
    U_t = U.cols(0, r - 1);
    V_t = V.cols(0, r - 1);
    S_t = diagmat(s.subvec(0, r - 1));
    tilde_N_test_t = U_t * S_t * V_t.t();
    
    double error = 0.0;
    for (uword i = 0; i < dim; i ++){
      N_t(i, i) = tilde_N_test_t(i, i);
      error += pow(tilde_N_test_t(i, i), 2.0);
    }
    
    if (abs(error - approx) > vartol){
      t += 1;
      approx = error;
    }
    else {
      break;
    }
  }
  return U_t;
}




// [[Rcpp::export]]
List tensor_hetero_pca_test_cpp(cube& Y, uvec r){

  mat U1;
  mat U2;
  mat U3;
  
  mat MY;
  mat MYT;
  
  uword tmax = 20;
  double tol = 1e-6;
  
  /* notice that the shape of MY is different from rTensor::k_unfold */
  MY = k_unfold_cpp(Y, 1);
  MYT = MY.t() * MY;
  U1 = hetero_pca_test_cpp(MYT, r.at(0), tmax, tol);
  
  MY = k_unfold_cpp(Y, 2);
  MYT = MY.t() * MY;
  U2 = hetero_pca_test_cpp(MYT, r.at(1), tmax, tol);
  
  MY = k_unfold_cpp(Y, 3);
  MYT = MY.t() * MY;
  U3 = hetero_pca_test_cpp(MYT, r.at(2), tmax, tol);
  
  return List::create( 
    _["u1"] = U1, 
    _["u2"] = U2, 
    _["u3"] = U3
  );
}


// [[Rcpp::export]]
cube ttm_cpp(cube& M, mat& U, uword k){
  cube res;
  
  switch (k){
  case 1:{
    res.zeros(U.n_rows, M.n_cols, M.n_slices);
    for (uword i = 0; i < U.n_rows; i++){
      for (uword j = 0; j < M.n_rows; j++){
        res.row(i) += M.row(j) * U(i, j);
      }
    }
  }
    break;
  case 2:{
    res.zeros(M.n_rows, U.n_rows, M.n_slices);
    for (uword i = 0; i < U.n_rows; i++){
      for (uword j = 0; j < M.n_cols; j++){
        res.col(i) += M.col(j) * U(i, j);
      }
    }
  }
    break;
  case 3:{
    res.zeros(M.n_rows, M.n_cols, U.n_rows);
    for (uword i = 0; i < U.n_rows; i++){
      for (uword j = 0; j < M.n_slices; j++){
        res.slice(i) += M.slice(j) * U(i, j);
      }
    }
  }
    break;
  }
  
  return res;
}



// [[Rcpp::export]]
cube hetero_pca_estimate_cpp(cube& Y, uvec r_hat){
  List U_hat = tensor_hetero_pca_test_cpp(Y, r_hat);
  mat U1 = U_hat["u1"];
  mat P_U1 =  U1 * U1.t();
  
  mat U2 = U_hat["u2"];
  mat P_U2 =  U2 * U2.t();
  
  mat U3 = U_hat["u3"];
  mat P_U3 =  U3 * U3.t();
  
  cube tmp = ttm_cpp(Y, P_U1, 1);
  tmp = ttm_cpp(tmp, P_U2, 2);
  cube Y_hat = ttm_cpp(tmp, P_U3, 3); 

  Y_hat.clamp(0.0, 1.0);  
  return Y_hat;
}

// // [[Rcpp::export]]
// cube hetero_pca_estimate_cpp(cube& Y, uvec r){
//   mat U1;
//   mat U2;
//   mat U3;
  
//   mat MY;
//   mat MYT;
  
//   uword tmax = 20;
//   double tol = 1e-6;
  
//   MY = k_unfold_cpp(Y, 1);
//   MYT = MY.t() * MY;
//   U1 = hetero_pca_test_cpp(MYT, r.at(0), tmax, tol);
  
//   MY = k_unfold_cpp(Y, 2);
//   MYT = MY.t() * MY;
//   U2 = hetero_pca_test_cpp(MYT, r.at(1), tmax, tol);
  
//   MY = k_unfold_cpp(Y, 3);
//   MYT = MY.t() * MY;
//   U3 = hetero_pca_test_cpp(MYT, r.at(2), tmax, tol);
  
//   mat P_U1 =  U1 * U1.t();
  
//   mat P_U2 =  U2 * U2.t();
  
//   mat P_U3 =  U3 * U3.t();
  
//   cube tmp = ttm_cpp(Y, P_U1, 1);
//   tmp = ttm_cpp(tmp, P_U2, 2);
//   cube Y_hat = ttm_cpp(tmp, P_U3, 3); 
  
//   Y_hat.clamp(0.0, 1.0);  
//   return Y_hat;
// }




// [[Rcpp::export]]
List svd_cpp(mat X){
  mat U;
  vec s;
  mat V;
  
  svd(U,s,V,X);
  
  return List::create( 
    _["u"]  = U, 
    _["s"]  = s, 
    _["v"] = V
  );
}








// [[Rcpp::export]]
inline urowvec shape (const Cube<double>& x) 
{ 
  return { x.n_rows, x.n_cols, x.n_slices };
}


// [[Rcpp::export]]
uword test (uvec& a)
{
  uword x = a[0];
  return x;
}


// [[Rcpp::export]]
Col<double> reshape_cpp (Mat<double>& a)
{
  // const uword n1 = a.n_rows;
  // const uword n2 = a.n_cols;
  Col<double> x = vectorise(a, 0);
  return x;
}


// [[Rcpp::export]]
Mat<double> check(Cube<double>& a){
  Mat<double> x = a.row(0);
  return x;
}

// [[Rcpp::export]]
Row<double> check_copy(Cube<double>& a){
  Cube<double> x = cube(a);
  x(0,0,0) = 100.0;
  double i = a(0,0,0);
  
  a(0,0,0) = 1.0;
  x = a;
  x(0,0,0) = 100.0;
  double j = a(0,0,0);
  
  // a(0,0,0) = 1.0;
  // x = cube(a, a.n_rows, a.n_cols, a.n_slices, false);
  // x(0,0,0) = 100.0;
  // double k = a(0,0,0);
  
  return {i, j};
}



// [[Rcpp::export]]
cube check_tensor_list(field<cube> A_list, double h_kernel){
  cube A = A_list[0];
  cube B = A_list[1];
  
  return A + B;
}      


// [[Rcpp::export]]
cube tensor_sum(field<cube> A_list, double h_kernel){
  cube A = A_list(0);

  const uword n1 = A.n_rows;
  const uword n2 = A.n_cols;
  const uword L = A.n_slices;
  const uword TT = A_list.n_elem;

  // const uword C_M = 20;
  // const double M_up = 1e3;
  // 
  // double n_left;
  // double n_right;

  cube A_sum_left;
  cube A_sum_right;

  mat Sigma = diagmat(vec(L, fill::value(1.0 / h_kernel)));

  vec D_K_t_max_rescale = vec(TT, fill::zeros);


  A_sum_left.zeros(n1, n2, L);
  for (uword i = 0; i< TT; i++){
    A_sum_left += A_list(i);
  }
  return A_sum_left;
}
      


// [[Rcpp::export]]
Cube<double> permute (Cube<double>& a, const uvec& order)
{
  uword idx1 = order[0];
  uword idx2 = order[1];
  uword idx3 = order[2];
  
  urowvec dimension = shape(a);
  
  uword rows = dimension[idx1 - 1];
  uword cols = dimension[idx2 - 1];
  uword slis = dimension[idx3 - 1];
  
  Cube<double> output = cube(rows, cols, slis);
  
  uword perm = idx1*100 + idx2*10 + idx3;
  
  switch (perm)
  {
  case 123:
  {
    output = a; // identity
  }
    break;
  case 132:
  {
    for (uword c = 0; c < a.n_cols; ++c)
      for (uword r = 0; r < a.n_rows; ++r)
        for (uword s = 0; s < a.n_slices; ++s)
          output(r, s, c) = a(r, c, s);
  }
    break;
  case 213:
  {
    for (uword c = 0; c < a.n_cols; ++c)
      for (uword r = 0; r < a.n_rows; ++r)
        for (uword s = 0; s < a.n_slices; ++s)
          output(c, r, s) = a(r, c, s);
  }
    break;
  case 231:
  {
    for (uword c = 0; c < a.n_cols; ++c)
      for (uword r = 0; r < a.n_rows; ++r)
        for (uword s = 0; s < a.n_slices; ++s)
          output(c, s, r) = a(r, c, s);
  }
    break;
  case 312:
  {
    for (uword c = 0; c < a.n_cols; ++c)
      for (uword r = 0; r < a.n_rows; ++r)
        for (uword s = 0; s < a.n_slices; ++s)
          output(s, r, c) = a(r, c, s);
  }
    break;
  case 321:
  {
    for (uword c = 0; c < a.n_cols; ++c)
      for (uword r = 0; r < a.n_rows; ++r)
        for (uword s = 0; s < a.n_slices; ++s)
          output(s, c, r) = a(r, c, s);
  }
    break;
  }
  
  return output;
}




// [[Rcpp::export]]
mat vec2mat(vec& x, const uword r, const uword c){
  mat m = mat(x);
  m.reshape(r, c);
  return m;
}

// [[Rcpp::export]]
mat vec2mat_manual(vec& x, const uword r, const uword c){
  mat m = mat(r, c);
  for(uword i = 0; i < r; i++){
    for(uword j = 0; j < c; j++){
      m(i, j) = x(i + j * r);
    }
  }
  return m;
}


// [[Rcpp::export]]
cube fold_cpp(mat& x, const uword n1, const uword n2, const uword n3){
  cube res = cube(n1, n2, n3);
  for(uword k = 0; k < n1; k++){
    for(uword i = 0; i < n2; i++){
      for(uword j = 0; j < n3; j++){
        res(k, i, j) = x(k, i + j * n2);
      }
    }
  }
  return res;
}




// [[Rcpp::export]]
cube uase_cpp(cube& a, uword r){
  
  mat Y = k_unfold_cpp(a, 1);
  
  mat U;
  vec s;
  mat V;
  
  svd(U, s, V, Y);
  
  r = std::min(r, s.size());
  
  // mat U_t = mat(Y.n_rows, r);
  // mat V_t = mat(Y.n_cols, r);
  // mat S_t = mat(r, r);
  
  mat U_t = U.cols(0, r - 1);
  mat V_t = V.cols(0, r - 1);
  mat S_t = diagmat(s.subvec(0, r - 1));
  
  mat P_uase =  V_t * S_t * U_t.t();

  P_uase.clamp(0.0, 1.0);
  
  cube Pt_uase = cube(a.n_rows, a.n_cols, a.n_slices);
  Pt_uase = fold_cpp(P_uase, a.n_rows, a.n_cols, a.n_slices);
  
  return Pt_uase;
  
}
