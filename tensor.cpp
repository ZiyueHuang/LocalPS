#include <iostream>

typedef unsigned index_t;

template<int dimension>
struct Shape {
  static const int kDimension = dimension;
  static const int kSubdim = dimension - 1;
  index_t shape_[kDimension];

  Shape(void) {}

  Shape(const Shape<kDimension> &s) {
    for (int i = 0; i < kDimension; ++i) {
      this->shape_[i] = s[i];
    }
  }
  
  index_t &operator[](index_t idx) {
    return shape_[idx];
  }
  
  const index_t &operator[](index_t idx) const {
    return shape_[idx];
  }
  
  bool operator==(const Shape<kDimension> &s) const {
    for (int i = 0; i < kDimension; ++i) {
      if (s.shape_[i] != this->shape_[i]) return false;
    }
    return true;
  }
  
  bool operator!=(const Shape<kDimension> &s) const {
    return !(*this == s);
  } 
  
  size_t Size(void) const {
    size_t size = this->shape_[0];
    for (int i = 1; i < kDimension; ++i) {
      size *= this->shape_[i];
    }
    return size;
  }

  Shape<kSubdim> SubShape(void) const {
    Shape<kSubdim> s;
    for (int i = 0; i < kSubdim; ++i) {
      s.shape_[i] = this->shape_[i + 1];
    }
    return s;
  }
    
  template<int dim>
  friend std::ostream &operator<<(std::ostream &os, const Shape<dim> &shape);
};


template<int ndim>
inline std::ostream &operator<<(std::ostream &os, const Shape<ndim> &shape) { 
  os << '(';
  for (int i = 0; i < ndim; ++i) {
    if (i != 0) os << ',';
    os << shape[i];
  }
  if (ndim == 1) os << ',';
  os << ')';
  return os;
}


template<int dimension, typename DType>
struct Tensor {
  static const int  kSubdim = dimension - 1;
  DType *dptr_;
  Shape<dimension> shape_;

  Tensor(void) : dptr_(nullptr) {}

  Tensor(const Shape<dimension> &shape)
      : shape_(shape), dptr_(nullptr) {}

  Tensor(DType *dptr, const Shape<dimension> &shape)
      : dptr_(dptr), shape_(shape) {}

  index_t size(index_t idx) const {
    return shape_[idx];
  }

  Tensor<kSubdim, DType> operator[](index_t idx) const {
    size_t sub_size = 1;
    for (int i = 1; i < dimension; i++) sub_size *= shape_[i];
    return Tensor<kSubdim, DType>(dptr_ + sub_size * idx,
                                          shape_.SubShape());
  }
  
  inline Tensor<dimension, DType> &
  operator=(const Tensor<dimension, DType> &exp) {
    dptr_ = exp.dptr_;
    shape_ = exp.shape_;
    return *this;
  }  
};


template<typename DType>
struct Tensor<1, DType> {
  DType *dptr_;
  Shape<1> shape_;
  
  Tensor(void) : dptr_(nullptr) {}

  Tensor(const Shape<1> &shape)
      : shape_(shape), dptr_(nullptr) {}

  Tensor(DType *dptr, const Shape<1> &shape)
      : dptr_(dptr), shape_(shape) {}

  index_t size(index_t i) const {
    return shape_[0];
  }

  DType &operator[](index_t idx) {
    return dptr_[idx];
  }

  const DType &operator[](index_t idx) const {
    return dptr_[idx];
  }

  inline Tensor<1, DType> &
  operator=(const Tensor<1, DType> &exp) {
    dptr_ = exp.dptr_;
    shape_ = exp.shape_;
    return *this;
  }  
};
