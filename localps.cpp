#include <map>
#include <utility>
#include <string>
#include <vector>
#include <thread>
#include <cassert>
#include <omp.h>
#include <string.h>
#include "./tensor.cpp"
#include "./utils.cpp"


template<int dim, typename DType>
void Copy(Tensor<dim, DType> dest, Tensor<dim, DType> src){
  assert(dest.shape_ == src.shape_);
  size_t size = dest.shape_.Size();
  memcpy(dest.dptr_, src.dptr_, sizeof(DType) * size);
}

template<int dim, typename DType>
void AllocSpace(Tensor<dim, DType>& src){
  size_t size = src.shape_.Size();
  src.dptr_ = new DType[size];
}

template<int dim, typename DType>
void FreeSpace(Tensor<dim, DType>& src){
  delete[] src.dptr_;
  src.dptr_ = nullptr;
}


template<typename DType>
class LocalModel {
 public:
  LocalModel(void) {
    init_end = 0;
    destroy_signal = false;
  }

  ~LocalModel(void) {
    this->Destroy();
  }

  inline void Destroy(void) {
    if (init_end != 0) {
      destroy_signal = true;
      push_queue.Abort();      
      pull_queue.Abort();
      
      push_thread.join();      
      pull_thread.join();
            
      push_map.Destroy();      
      pull_map.Destroy();

      init_end = 0;
    }    
  }
  
  void PullWait(int key, int tid) {
    PullEntry &e = pull_map.GetRef(key);
    assert(e.wait.size() == num_thread);
    PullWaitRecord &w = e.wait[tid];
    if (!w.finished) {
      std::unique_lock<std::mutex> wait_lock(wait_mutex);
      w.nwait += 1;
      wait_cond.wait(wait_lock, [&]{return w.finished;});
      
      w.nwait -= 1;
      assert(w.nwait >= 0);
    }
  }

  void Init(const int num_thread) {
    assert(init_end == 0);
    this->num_thread = num_thread;
    destroy_signal = false;   
    push_thread = std::thread(PushGlobalThread, this);    
    pull_thread = std::thread(PullGlobalThread, this);      
    this->init_end = 1;
  }

  enum LocalOp {
    kSum = 0
  };

  void InitKey(int key, Shape<2> shape) {
    this->InitPullMap(key);
    this->InitPushMap(key, shape);
  }

  void Push(Tensor<2, DType> data, int key, int tid) {
    PullEntry &e = pull_map.GetRef(key);
    e.req[tid].ready = false;
     
    push_queue.Push(PullTask(data, key, tid));
  }

  void PullReq(Tensor<2, DType> data, int key, int tid) {
    PullEntry &e = pull_map.GetRef(key);
    assert(e.req.size() == num_thread);
    assert(e.wait.size() == num_thread);
    {
      std::lock_guard<std::mutex> wait_lock(wait_mutex);
      e.wait[tid].finished = false;
    }
    PullReqRecord &r = e.req[tid];
    {
      std::lock_guard<std::mutex> request_lock(request_mutex);
      r.dest = data;
      assert(!r.pending);
      if (e.req[tid].ready) {      
        pull_queue.Push(std::make_pair(key, tid));      
      } else {
        r.pending = true;
      }
    }

  }
  
  void PullReady(Tensor<2, DType> data, int key) {
    PullEntry &e = pull_map.GetRef(key);
    assert(e.req.size() == num_thread);

    std::lock_guard<std::mutex> lock(request_mutex);
    e.src = data;
    for (index_t i = 0; i < e.req.size(); ++i) {
      e.req[i].ready = true;
      if (e.req[i].pending) {        
        pull_queue.Push(std::make_pair(key, i));
        e.req[i].pending = false;
      }
    }
  }
  

  void HandlePushFinish(Tensor<3, DType> data, int key) {
    LocalOp op = kSum;
    
    switch (op) {
      case kSum: {
        this->ReduceSum(data);
        this->PullReady(data[0], key);
        return;
      }
      default: assert(false);
    }
  }


  inline void ReduceSum(Tensor<3, DType> data) {    
    for (index_t i = 1; i < data.size(0); ++i) {
      for (index_t j = 0; j < data.size(1); ++j) {
        for (index_t k = 0; k < data.size(2); ++k) {
          data[0][j][k] += data[i][j][k];      
        }      
      }      
    }
  }


  struct PullTask {
    Tensor<2, DType> data;
    int key;    
    int tid;
    PullTask(void) {}
    PullTask(Tensor<2, DType> data, int key, int tid)
        : data(data), key(key), tid(tid) {}
  };

  struct PushEntry {
    Tensor<4, DType> data;   
    std::vector<bool> copied;
    int num_copied;
    int copyin_version;
    
    PushEntry(void) : copyin_version(0) {
    }
    ~PushEntry(void) {
      if (data.dptr_ != nullptr) {        
        FreeSpace(data);                
      }
    }
    inline void Init(int num_thread, Shape<2> shape) {
      data.shape_ = Shape<4>();
      data.shape_[0] = 2;
      data.shape_[1] = num_thread;
      data.shape_[2] = shape[0];
      data.shape_[3] = shape[1];
     
      AllocSpace(data);
      
      num_copied = 0;
      copied.resize(num_thread, false);
    }
  };

  struct PullReqRecord {
    bool ready;
    bool pending;
    Tensor<2, DType> dest;

    PullReqRecord(void) : ready(false), pending(false) {
    }
  };

  struct PullWaitRecord {
    int nwait;
    bool finished;
    PullWaitRecord(void): nwait(0), finished(true) {
    }
  };

  struct PullEntry {
    Tensor<2, DType> src;
    std::vector<PullReqRecord> req;
    std::vector<PullWaitRecord> wait;
    PullEntry(void) {
    }
  };

  bool destroy_signal;

  int num_thread;

  ThreadSafeQueue<PullTask> push_queue;
  std::thread push_thread;
  std::mutex push_mutex;
  ThreadSafeMap<PushEntry> push_map;

  ThreadSafeQueue<std::pair<int, int>> pull_queue;
  ThreadSafeMap<PullEntry> pull_map;
  std::thread pull_thread;

  std::mutex request_mutex;
  std::mutex wait_mutex;
  std::condition_variable wait_cond;
  
  int init_end;
  
  inline void PushProc(void) {
    while (!destroy_signal) {
      PullTask tsk;
      if (push_queue.Pop(tsk)) {
        const int tid = tsk.tid;
        PushEntry &e = push_map.GetRef(tsk.key);
        assert(e.data[0][0].shape_ == tsk.data.shape_);
        assert(!e.copied[tid]);

        Copy(e.data[e.copyin_version][tid], tsk.data);
        std::unique_lock<std::mutex> push_lock(push_mutex);
        e.copied[tid] = true;
        e.num_copied += 1;
        int cp_version = e.copyin_version;
        bool push_finish = e.num_copied == num_thread;
        if (push_finish) {
          e.copyin_version = (e.copyin_version + 1) % e.data.size(0);
          std::fill(e.copied.begin(), e.copied.end(), false);
          e.num_copied = 0;
        }
        push_lock.unlock();

        if (push_finish) {
          this->HandlePushFinish(e.data[cp_version], tsk.key);
        }
      } else {
        assert(destroy_signal);
      }
    }
  } 
  
  
  inline void PullProc(void) {
    while (!destroy_signal) {
      std::pair<int, int> tsk;
      if (pull_queue.Pop(tsk)) {
        const int key = tsk.first;
        const int tid = tsk.second;
        PullEntry &e = pull_map.GetRef(key);
        
        assert(e.req.size() == num_thread);
        PullReqRecord &r = e.req[tid];

        Copy(r.dest, e.src);
      
        assert(e.wait.size() == num_thread);
        PullWaitRecord &w = e.wait[tid];

        std::unique_lock<std::mutex> wait_lock(wait_mutex);
        w.finished = true;
        wait_lock.unlock();

        wait_cond.notify_all();
                  
      } else {
        assert(destroy_signal);
      }
    }
  }
  
  static void PullGlobalThread(void *ptr) {
    (static_cast<LocalModel<float>*>(ptr))->PullProc();
  }

  static void PushGlobalThread(void *ptr) {
    (static_cast<LocalModel<float>*>(ptr))->PushProc();
  }

  inline void InitPullMap(int key) {
    pull_map.Init(key);
    PullEntry &e = pull_map.GetRef(key);
    {
      std::lock_guard<std::mutex> request_lock(request_mutex);
      if (e.req.size() == 0) {
        e.req.resize(num_thread, PullReqRecord());
      }
    }
    {
      std::lock_guard<std::mutex> wait_lock(wait_mutex);
      if (e.wait.size() == 0) {
        e.wait.resize(num_thread, PullWaitRecord());
      }
    }
    
  }
  inline void InitPushMap(int key, Shape<2> shape) {
    push_map.Init(key);
    PushEntry &e = push_map.GetRef(key);
    std::lock_guard<std::mutex> push_lock(push_mutex);
    if (e.copied.size() == 0) {
      e.Init(num_thread, shape);
    }
  }

};


int main(){
  LocalModel<float> *localps = new LocalModel<float>();
  const int nthread = 2;
  localps->Init(nthread);

  Shape<2> s; s[0]=2; s[1]=2;

  Tensor<2, float> t1(s);
  Tensor<2, float> t2(s);
  Tensor<2, float> t3(s);
  Tensor<2, float> t4(s);
  Tensor<2, float> t[2], tt[2];

  AllocSpace(t1);
  AllocSpace(t2);
  AllocSpace(t3);
  AllocSpace(t4);

  t1[0][0]=1; t1[0][1]=1; t1[1][0]=1; t1[1][1]=1;
  t2[0][0]=2; t2[0][1]=2; t2[1][0]=2; t2[1][1]=2;
  t[0]=t1; t[1]=t2;
  tt[0]=t3; tt[1]=t4;

  localps->InitKey(0, s);
  
  #pragma omp parallel num_threads(2)
  {
    int tid = omp_get_thread_num();
    localps->Push(t[tid], 0, tid);
    localps->PullReq(tt[tid], 0, tid);
    localps->PullWait(0, tid);
  }

  std::cout<<t3[1][0]<<std::endl;

  FreeSpace(t1);
  FreeSpace(t2);
  FreeSpace(t3);
  FreeSpace(t4);

  delete localps;
  return 0;
}
