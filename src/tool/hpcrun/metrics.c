// -*-Mode: C++;-*- // technically C99

// * BeginRiceCopyright *****************************************************
//
// $HeadURL$
// $Id$
//
// --------------------------------------------------------------------------
// Part of HPCToolkit (hpctoolkit.org)
//
// Information about sources of support for research and development of
// HPCToolkit is at 'hpctoolkit.org' and in 'README.Acknowledgments'.
// --------------------------------------------------------------------------
//
// Copyright ((c)) 2002-2018, Rice University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of Rice University (RICE) nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// This software is provided by RICE and contributors "as is" and any
// express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular
// purpose are disclaimed. In no event shall RICE or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or
// business interruption) however caused and on any theory of liability,
// whether in contract, strict liability, or tort (including negligence
// or otherwise) arising in any way out of the use of this software, even
// if advised of the possibility of such damage.
//
// ******************************************************* EndRiceCopyright *

//************************* System Include Files ****************************

#include <stdbool.h>
#include <stdlib.h>

#include <assert.h>

//*************************** User Include Files ****************************

#include "monitor.h"
#include "metrics.h"

#include <memory/hpcrun-malloc.h>

#include <messages/messages.h>

#include <cct/cct.h>

#include <lib/prof-lean/hpcio.h>
#include <lib/prof-lean/hpcfmt.h>
#include <lib/prof-lean/hpcrun-fmt.h>

//*************************** Concrete Data Types ***************************

//
// following typedef accomplishes
// metric_set_t* == array of metric values, but it does this abstractly
// so that clients of the metric datatype must use the interface.
//
struct  metric_set_t {
  hpcrun_metricVal_t v1;
};

//*************************** Local Data **************************

// some sample sources will pre-allocate some metrics ...
static metric_desc_list_t* pre_alloc = NULL;

//
// To accomodate block sparse representation,
// use 'kinds' == dense subarrays of metrics
//
// Sample use:
// Std metrics = 1 class,
// CUDA metrics = another class
//
// To use the mechanism, call hpcrun_metrics_new_kind.
// Then each call to hpcrun_new_metric will yield a slot in the
// new metric kind subarray.
//
// For complicated metric assignment, hpcrun_metrics_switch_kind(kind),
// and hpcrun_new_metric_of_kind(kind) enable fine-grain control
// of metric sloc allocation
//
// Default case is 1 kind.
//
// Future expansion to permit different strategies is possible, but
// unimplemented at this time

struct kind_info_t {
  int idx;     // current index in kind
  bool has_set_max;
  kind_info_t* link; // all kinds linked together in singly linked list
  // metric_tbl serves 2 purposes:
  //    1) mapping from metric_id ==> metric desc, so that samples will increment correct metric slot
  //       in the cct node
  //    2) metric info is written out in metric_tbl form
  //
  metric_desc_p_tbl_t metric_tbl;
// Dense array holding "0" values for nodes with no metrics
  hpcrun_metricVal_t *null_metrics;
// information about tracked metrics
  metric_desc_list_t* metric_data;
// local table of metric update functions
// (indexed by metric id)
  metric_upd_proc_t** metric_proc_tbl;

};

static kind_info_t kinds = {.idx = 0, .has_set_max = 0, .link = NULL };
static kind_info_t* current_kind = &kinds;
static int num_kind_metrics;

kind_info_t*
hpcrun_metrics_new_kind(void)
{
  kind_info_t* rv = (kind_info_t*) hpcrun_malloc(sizeof(kind_info_t));
  *rv = (kind_info_t) {.idx = 0, .has_set_max = 0, .link = NULL};
  current_kind->link = rv;
  current_kind = rv;
  return rv;
}

void
hpcrun_metrics_switch_kind(kind_info_t* kind)
{
  current_kind = kind;
}

//

typedef struct metric_data_list_t {
  struct metric_data_list_t* next;
  kind_info_t *kind;
  metric_set_t *metrics;
} metric_data_list_t;

static metric_proc_map_t* proc_map = NULL;


//***************************************************************************
//  Local functions
//***************************************************************************


//***************************************************************************
//  Interface functions
//***************************************************************************

bool
hpcrun_metrics_finalized()
{
  return current_kind->has_set_max;
}


void
hpcrun_pre_allocate_metrics(size_t num)
{
  if (current_kind->has_set_max) {
    return;
  }
  for(int i=0; i < num; i++){
    metric_desc_list_t* n = (metric_desc_list_t*) hpcrun_malloc(sizeof(metric_desc_list_t));
    n->next = pre_alloc;
    pre_alloc = n;
  }
  // NOTE: actual metric count not incremented until a new metric is requested
}


int
hpcrun_get_num_kind_metrics()
{
  return (num_kind_metrics);
}

//
// first call to get_num_metrics will finalize
// the metric info table
//
int
hpcrun_get_num_metrics()
{
  int n_metrics = current_kind->idx;
  //
  // create id->descriptor table, metric_tbl, and metric_proc tbl
  //
  if (!current_kind->has_set_max) {
    current_kind->metric_tbl.len = n_metrics;
    current_kind->metric_tbl.lst = hpcrun_malloc(current_kind->idx * sizeof(metric_desc_t*));
    for(metric_desc_list_t* l = current_kind->metric_data; l; l = l->next){
      TMSG(METRICS_FINALIZE,"metric_tbl[%d] = %s", l->id, l->val.name);
      current_kind->metric_tbl.lst[l->id] = &(l->val);
    }
    current_kind->metric_proc_tbl = (metric_upd_proc_t**) hpcrun_malloc(n_metrics * sizeof(metric_upd_proc_t*));
    
    for(metric_proc_map_t* l = proc_map; l; l = l->next) {
    //    for(metric_proc_map_t* l = proc_map; l; l = l->next) {
      TMSG(METRICS_FINALIZE, "metric_proc[%d] = %p", l->id, l->proc);
      current_kind->metric_proc_tbl[l->id] = l->proc;
    }
  // *** TEMPORARY ***
  // *** create a "NULL METRICS" dense array for use with
  // *** metric set dense copy

    current_kind->null_metrics = hpcrun_malloc(n_metrics * sizeof(hpcrun_metricVal_t));
    for (int i = 0; i < n_metrics; i++) {
      current_kind->null_metrics[i].bits = 0;
    }
    num_kind_metrics += n_metrics;
  }
  current_kind->has_set_max = true;

  return n_metrics;
}

// Finalize metrics

void hpcrun_finalize_metrics() 
{
  hpcrun_get_num_metrics();
}

metric_desc_t*
hpcrun_id2metric(int metric_id)
{
  int n_metrics = hpcrun_get_num_metrics(); 
  if ((0 <= metric_id) && (metric_id < n_metrics)) {
    return current_kind->metric_tbl.lst[metric_id];
  }
  return NULL;
}


metric_desc_p_tbl_t*
hpcrun_get_metric_tbl(kind_info_t **curr)
{
  if (*curr == NULL)
    *curr = &kinds;
  else
    *curr = (*curr)->link;

  return &(*curr)->metric_tbl;
}

metric_desc_list_t*
hpcrun_get_metric_desc_data()
{
  return current_kind->metric_data;
}


// *** FIXME? metric finalization may not be necessary
// when metric set repr changes
//
metric_upd_proc_t*
hpcrun_get_metric_proc(int metric_id)
{
  int n_metrics = hpcrun_get_num_metrics(); // ensure that metrics are finalized

  if ((0 <= metric_id) && (metric_id < n_metrics)) {
    return current_kind->metric_proc_tbl[metric_id];
  }

  return NULL;
}

metric_data_list_t *hpcrun_new_metric_data_list()
{
  metric_data_list_t *curr = hpcrun_malloc(sizeof(metric_data_list_t));
  curr->kind = current_kind;
  curr->metrics = hpcrun_metric_set_new();
  curr->next = NULL;
  return curr;
}

//
// Allocate new metric of a particular kind
//
static int
hpcrun_new_metric_of_kind(kind_info_t* kind)
{
  if (kind->has_set_max) {
    return -1;
  }

  metric_desc_list_t* n = NULL;

  // if there are pre-allocated metrics, use them
  if (pre_alloc) {
    n = pre_alloc;
    pre_alloc = pre_alloc->next;
  }
  else {
    n = (metric_desc_list_t*) hpcrun_malloc(sizeof(metric_desc_list_t));
  }
  n->next = kind->metric_data;
  kind->metric_data = n;
  kind->metric_data->id   = kind->idx++;

  //
  // No preallocation for metric_proc tbl
  //
  metric_proc_map_t* m = (metric_proc_map_t*) hpcrun_malloc(sizeof(metric_proc_map_t));
  m->next = proc_map;
  m->id   = kind->metric_data->id;
  m->proc = (metric_upd_proc_t*) NULL;
  proc_map = m;
  
  return m->id;
}

static int
hpcrun_new_metric(void)
{
  return hpcrun_new_metric_of_kind(current_kind);
}

static void
hpcrun_set_metric_info_w_fn(int metric_id, const char* name,
			    MetricFlags_ValFmt_t valFmt, size_t period,
			    metric_upd_proc_t upd_fn, metric_desc_properties_t prop)
{
  if (current_kind->has_set_max) {
    return;
  }

  metric_desc_t* mdesc = NULL;
  for (metric_desc_list_t* l = current_kind->metric_data; l; l = l->next) {
    if (l->id == metric_id) {
      mdesc = &(l->val);
      break;
    }
  }
  TMSG(METRICS,"id = %d, name = %s, flags = %d, period = %d", metric_id, name, valFmt, period);
  if (! mdesc) {
    EMSG("Metric id is NULL (likely unallocated)");
    monitor_real_abort();
  }
  if (name == NULL) {
    EMSG("Must supply a name for metric");
    monitor_real_abort();
  }

  *mdesc = metricDesc_NULL;
  mdesc->flags = hpcrun_metricFlags_NULL;

  mdesc->name = (char*) name;
  mdesc->description = (char*) name; // TODO
  mdesc->period = period;
  mdesc->flags.fields.ty     = MetricFlags_Ty_Raw; // FIXME
  mdesc->flags.fields.valFmt = valFmt;
  mdesc->formula = NULL;
  mdesc->format = NULL;
  mdesc->properties = prop;

  //
  // manage metric proc mapping
  //
  for (metric_proc_map_t* l = proc_map; l; l = l->next){
    if (l->id == metric_id){
      l->proc = upd_fn;
      break;
    }
  }
}

int
hpcrun_set_new_metric_info_w_fn(const char* name,
				MetricFlags_ValFmt_t valFmt, size_t period,
				metric_upd_proc_t upd_fn, metric_desc_properties_t prop)
{
  int metric_id = hpcrun_new_metric();
  hpcrun_set_metric_info_w_fn(metric_id, name, valFmt, period, upd_fn, prop);
  return metric_id;
}


static void
hpcrun_set_metric_info_and_period(int metric_id, const char* name,
				  MetricFlags_ValFmt_t valFmt, size_t period, metric_desc_properties_t prop)
{
  return  hpcrun_set_metric_info_w_fn(metric_id, name, valFmt, period,
			      hpcrun_metric_std_inc, prop);
}


int
hpcrun_set_new_metric_info_and_period(const char* name,
				      MetricFlags_ValFmt_t valFmt, size_t period, metric_desc_properties_t prop)
{
  int metric_id = hpcrun_new_metric();
  hpcrun_set_metric_info_w_fn(metric_id, name, valFmt, period,
			      hpcrun_metric_std_inc, prop);
  return metric_id;
}


int
hpcrun_set_new_metric_info(const char* name)
{
  int metric_id = hpcrun_new_metric();
  hpcrun_set_metric_info_and_period(metric_id, name, MetricFlags_ValFmt_Int, 1, metric_property_none);
  return metric_id;
}

//
// utility routine to make an Async metric with period 1
//
void
hpcrun_set_metric_info(int metric_id, const char* name)
{
  hpcrun_set_metric_info_and_period(metric_id, name,
				    MetricFlags_ValFmt_Int, 1, metric_property_none);
}


//
// hpcrun_set_metric_name function is primarily used by
// synchronous sample sources that need to name the metrics
// after they have been allocated.
//
// makes a call to hpcrun_get_num_metrics just to ensure that
// metric_tbl has been finalized.
//
void
hpcrun_set_metric_name(int metric_id, char* name)
{
  int n_metrics = hpcrun_get_num_metrics();

  if ((0 <= metric_id) && (metric_id < n_metrics)) {
    current_kind->metric_tbl.lst[metric_id]->name        = name;
    current_kind->metric_tbl.lst[metric_id]->description = name;
  }
}

//
// Metric set interface
//



metric_set_t*
hpcrun_metric_set_new(void)
{
  int n_metrics = hpcrun_get_num_metrics();
  return hpcrun_malloc(n_metrics * sizeof(hpcrun_metricVal_t));
}

//
// return an lvalue from metric_set_t*
//
cct_metric_data_t*
hpcrun_metric_set_loc(metric_set_t* s, int id)
{
  int n_metrics = hpcrun_get_num_metrics();
  if (s && (0 <= id) && (id < n_metrics)) {
    return &(s->v1) + id;
  }
  return NULL;
}


void
hpcrun_metric_std(int metric_id, metric_set_t* set,
		  char operation, hpcrun_metricVal_t val)
{
  metric_desc_t* minfo = current_kind->metric_tbl.lst[metric_id];
  if (!minfo) {
    return;
  }

  hpcrun_metricVal_t* loc = hpcrun_metric_set_loc(set, metric_id);
  switch (minfo->flags.fields.valFmt) {
    case MetricFlags_ValFmt_Int:
      if (operation == '+')
        loc->i += val.i; 
      else if (operation == '=')
        loc->i = val.i;
      break;
    case MetricFlags_ValFmt_Real:
      if (operation == '+')
        loc->r += val.r;
      else if (operation == '=')
        loc->r = val.r;
      break;
    default:
      assert(false);
  }
}
//
// replace the old value with the new value
//
void
hpcrun_metric_std_set(int metric_id, metric_set_t* set,
		      hpcrun_metricVal_t value)
{
  hpcrun_metric_std(metric_id, set, '=', value);
}

// increasing the value of metric
//
void
hpcrun_metric_std_inc(int metric_id, metric_set_t* set,
		      hpcrun_metricVal_t incr)
{
  hpcrun_metric_std(metric_id, set, '+', incr);
}

// add a metric of the current kind to a metric_data_list, if it's not already 
// there, and return the metric
metric_set_t*
hpcrun_add_current_metric(metric_data_list_t *rv)
{
  metric_data_list_t *curr;
  for (curr = rv; curr != NULL && curr->kind != current_kind; rv = curr, curr = curr->next);
  if (curr == NULL) {
    TMSG(CCT2METRICS, " -- Metric set was null, allocating new metric set");
    curr = rv->next = hpcrun_new_metric_data_list();
    if (curr == NULL)
      return NULL;
  }
  return curr->metrics;
}

// find a metric of the current kind to a metric_data_list, or null
metric_set_t*
hpcrun_find_current_metric(metric_data_list_t *rv)
{
  metric_data_list_t *curr;
  for (curr = rv; curr != NULL && curr->kind != current_kind; curr = curr->next);
  if (curr == NULL)
    return (NULL);
  return curr->metrics;
}


//
// copy a metric set
//
void
hpcrun_metric_set_dense_copy(cct_metric_data_t* dest,
			     metric_data_list_t* list,
			     int num_metrics)
{
  int offset = 0;
  kind_info_t *curr_k = &kinds;
  metric_data_list_t *curr;

  for (curr_k = &kinds; curr_k != NULL; curr_k = curr_k->link) {
    for (curr = list; curr != NULL && curr->kind != curr_k; curr = curr->next);
    metric_set_t* actual = curr ? curr->metrics : (metric_set_t*) curr_k->null_metrics;
    memcpy((char*) dest + offset, (char*) actual, curr_k->idx * sizeof(cct_metric_data_t));
    offset += curr_k->idx;
  }
}
