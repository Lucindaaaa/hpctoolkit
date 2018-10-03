#ifndef __MetricAccessor_hpp__
#define __MetricAccessor_hpp__

class MetricAccessor {
public:
  virtual ~MetricAccessor() {}; 
  virtual double &idx(int mId, int size = 0) = 0;
  virtual double c_idx(int mId) const = 0;
};

#endif

