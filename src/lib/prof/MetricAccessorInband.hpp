#ifndef __MetricAccessorInband_hpp__
#define __MetricAccessorInband_hpp__

#include "MetricAccessor.hpp"


class MetricAccessorInband : public MetricAccessor {
public:
  MetricAccessorInband(MetricAccessor *_mdata) : mdata(_mdata) {}
  virtual double &idx(int mId, int size = 0) {
    return mdata->idx(mId, size);
  }
  virtual double c_idx(int mId) const {
    return mdata->c_idx(mId);
  }
  virtual int idx_ge(int mId) const {
    return mId;
  }
  virtual bool empty(void) const {
    return mdata->empty();
  }
private:
  MetricAccessor *mdata;
};

#endif
