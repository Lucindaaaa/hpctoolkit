#ifndef CORE_PROFILE_TRACE_DATA_H
#define CORE_PROFILE_TRACE_DATA_H

#include <stdint.h>
#include <stdio.h>
#include <lib/prof-lean/hpcio-buffer.h>
#include <lib/prof-lean/hpcfmt.h> // for metric_aux_info_t

#include "epoch.h"
#include "cct2metrics.h"

enum perf_ksym_e {PERF_UNDEFINED, PERF_AVAILABLE, PERF_UNAVAILABLE} ;

#if 0 /*yumeng*/
typedef struct output_metric_id_offset_t{
  uint16_t id;
  uint64_t offset;
}output_metric_id_offset_t;

typedef struct cct_metrics_tid_sparse_data_t {
  hpcrun_metricVal_t* vals;
  int tid;
  output_metric_id_offset_t* metric_id_offsets;
  uint64_t* cct_offsets;
}cct_metrics_tid_sparse_data_t;
#endif 

typedef struct core_profile_trace_data_t {
  int id;
  // ----------------------------------------
  // epoch: loadmap + cct + cct_ctxt
  // ----------------------------------------
  epoch_t* epoch;

  //metrics: this is needed otherwise 
  //hpcprof does not pick them up
  cct2metrics_t* cct2metrics_map;

  //yumeng: for constructing cct_metricid_threads_values sparse matrix
  //cct_metrics_tid_sparse_data_t* sparse_metrics_values;

  // for metric scale (openmp uses)
  void (*scale_fn)(void*);
  // ----------------------------------------
  // tracing
  // ----------------------------------------
  uint64_t trace_min_time_us;
  uint64_t trace_max_time_us;

  // ----------------------------------------
  // IO support
  // ----------------------------------------
  FILE* hpcrun_file;
  void* trace_buffer;
  hpcio_outbuf_t *trace_outbuf;

  // ----------------------------------------
  // Perf support
  // ----------------------------------------

  metric_aux_info_t *perf_event_info;

} core_profile_trace_data_t;


#endif























