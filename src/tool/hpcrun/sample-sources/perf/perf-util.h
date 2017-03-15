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
// Copyright ((c)) 2002-2017, Rice University
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

#ifndef __PERF_UTIL_H__
#define __PERF_UTIL_H__

#include <linux/types.h>


#ifndef u32
typedef __u32 u32;
#endif


#ifndef u64
typedef __u64 u64;
#endif

// data from perf's mmap
typedef struct perf_mmap_data_s {
  struct perf_event_header header;
  u64    sample_id;  /* if PERF_SAMPLE_IDENTIFIER */
  u64    ip;         /* if PERF_SAMPLE_IP */
  u32    pid, tid;   /* if PERF_SAMPLE_TID */
  u64    time;       /* if PERF_SAMPLE_TIME */
  u64    addr;       /* if PERF_SAMPLE_ADDR */
  u64    id;         /* if PERF_SAMPLE_ID */
  u64    stream_id;  /* if PERF_SAMPLE_STREAM_ID */
  u32    cpu, res;   /* if PERF_SAMPLE_CPU */
  u64    period;     /* if PERF_SAMPLE_PERIOD */
                     /* if PERF_SAMPLE_READ */
  u64    nr;         /* if PERF_SAMPLE_CALLCHAIN */
  u64    *ips;       /* if PERF_SAMPLE_CALLCHAIN */
  u32    size;       /* if PERF_SAMPLE_RAW */
  char   *data;      /* if PERF_SAMPLE_RAW */
  /* if PERF_SAMPLE_BRANCH_STACK */
  
                     /* if PERF_SAMPLE_BRANCH_STACK */
  u64    abi;        /* if PERF_SAMPLE_REGS_USER */
  u64    *regs;
                     /* if PERF_SAMPLE_REGS_USER */
  u64    stack_size;             /* if PERF_SAMPLE_STACK_USER */
  char   *stack_data; /* if PERF_SAMPLE_STACK_USER */
  u64    stack_dyn_size;         /* if PERF_SAMPLE_STACK_USER &&
                                     size != 0 */
  u64    weight;     /* if PERF_SAMPLE_WEIGHT */
  u64    data_src;   /* if PERF_SAMPLE_DATA_SRC */
  u64    transaction;/* if PERF_SAMPLE_TRANSACTION */
  u64    intr_abi;        /* if PERF_SAMPLE_REGS_INTR */
  u64    *intr_regs;
                     /* if PERF_SAMPLE_REGS_INTR */
} perf_mmap_data_t;

#endif