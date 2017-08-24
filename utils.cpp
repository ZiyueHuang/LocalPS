#include<mutex>
#include<map>
#include<queue>
#include<omp.h>
#include <condition_variable>
#include<cassert>
#include<iostream>
#include<thread>

template<typename T>
class ThreadSafeQueue
{
  public:
    ThreadSafeQueue(){destroy=false;}
    void Push(const T &value);
    bool Empty();
    void Abort();
    bool Pop(T& popped_value);
  private:
    bool destroy;
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond;
};

template<typename T>
bool ThreadSafeQueue<T>::Empty() {
  std::unique_lock<std::mutex> lock(mutex_);
  return queue_.empty();
}

template<typename T>
void ThreadSafeQueue<T>::Abort() {
  destroy = true;
  cond.notify_all();
}

template<typename T>
void ThreadSafeQueue<T>::Push(const T& value) {
  std::unique_lock<std::mutex> lock(mutex_);
  queue_.push(value);
  lock.unlock();
  cond.notify_one();
}

template<typename T>
bool ThreadSafeQueue<T>::Pop(T& popped_value) {
  std::unique_lock<std::mutex> lock(mutex_);
  cond.wait(lock, [&]{return !queue_.empty() || destroy;});

  if (queue_.empty()) return false;

  popped_value = queue_.front();
  queue_.pop();
  return true;
}




template<typename T>
class ThreadSafeMap {
 public:
  inline void Destroy(void) {
    for (typename std::map<int, T*>::iterator
             it = map_.begin(); it != map_.end(); ++it) {
      delete it->second;
    }
  }

  inline T *Get(int key) {
    T *ret;
    std::lock_guard<std::mutex> lock(mutex_);
    typename std::map<int, T*>::const_iterator
        it = map_.find(key);
    if (it == map_.end() || it->first != key) {
      ret = nullptr;
    } else {
      ret = it->second;
    }
    return ret;
  }

  inline T &GetRef(int key) {
    T *ret = this->Get(key);
    assert(ret != nullptr);
    return *ret;
  }

  inline void Init(int key) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (map_.count(key) == 0) {
      map_[key] = new T();
    }
  }

 private:
  mutable std::mutex mutex_;
  std::map<int, T*> map_;
};

/*
class test{
public:
  static void push_thd(void *ptr){
    test *ppp = (static_cast<test*>(ptr));
    int i;
    while((ppp->p).Pop(i)) std::cout<<i<<std::endl;
    std::cout<<"o"<<std::endl;
    
  }
  void init(){
    thd = std::thread(push_thd, this);
  }
  ~test(){this->des();}
  void des(){
    p.Abort();
    thd.join();
  }
  void mypush(int j){
    p.Push(j);
  }
  std::thread thd;
  ThreadSafeQueue<int> p;
};


int main(){
  test *mytest = new test();
  mytest->init();
  #pragma omp parallel num_threads(3)
  {
    int tid = omp_get_thread_num();
    mytest->mypush(tid);
    
  }
  //mytest->des();
  delete mytest;
  return 0;
}
*/
