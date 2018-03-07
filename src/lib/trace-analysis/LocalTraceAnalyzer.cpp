// -*-Mode: C++;-*-

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
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or
// business interruption) however caused and on any theory of liability,
// whether in contract, strict liability, or tort (including negligence
// or otherwise) arising in any way out of the use of this software, even
// if advised of the possibility of such damage.
//
// ******************************************************* EndRiceCopyright *

/* 
 * File:   LocalTraceAnalyzer.cpp
 * Author: Lai Wei <lai.wei@rice.edu>
 * 
 * Created on March 6, 2018, 11:27 PM
 * 
 * Analyzes traces for a rank/thread and generates a summary temporal context tree.
 */

#include <stdio.h>
#include <string>
using std::string;

#include <lib/prof-lean/hpcrun-fmt.h>

#include "LocalTraceAnalyzer.hpp"

namespace TraceAnalysis {
  
  // Handles read from raw trace file.
  class TraceFileReader {
  public:
    TraceFileReader(const string filename, Time minTime) : minTime(minTime) {
      file = fopen(filename.c_str(),"r");
      if (file != NULL) {
        hpctrace_fmt_hdr_fread(&hdr, file);
      }
    }
    virtual ~TraceFileReader() {
      if (file != NULL) fclose(file);
    }

    // Read the next trace record in the trace file.
    // Return true when successful; false when error or at end of file.
    bool readNextTrace(hpctrace_fmt_datum_t* trace) {
      int ret = hpctrace_fmt_datum_fread(trace, hdr.flags, file);
      trace->time -= minTime;
      if (ret == HPCFMT_OK) return true;
      else return false;
    }

    const Time minTime;
    FILE* file;
    hpctrace_fmt_hdr_t hdr;
  };
  
  LocalTraceAnalyzer::LocalTraceAnalyzer(BinaryAnalyzer& binaryAnalyzer, 
          CCTVisitor& cctVisitor, string traceFileName, Time minTime) : 
          binaryAnalyzer(binaryAnalyzer), cctVisitor(cctVisitor),
          reader(new TraceFileReader(traceFileName, minTime)) {}

  LocalTraceAnalyzer::~LocalTraceAnalyzer() {
    delete reader;
  }
  
  void LocalTraceAnalyzer::analyze() {
    if (reader->file == NULL) return;
    
    hpctrace_fmt_datum_t trace;
    while (reader->readNextTrace(&trace)) {
      printf("cpid=0x%x, time=%s, dLCA=%u.\n", trace.cpId, 
              timeToString(trace.time).c_str(), trace.dLCA);
    }
  }
}

