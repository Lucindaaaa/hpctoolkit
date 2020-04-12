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
// Copyright ((c)) 2020, Rice University
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

#include "sparse.hpp"

#include "mpi-strings.h"
#include "lib/profile/util/log.hpp"

#include "lib/prof-lean/hpcrun-fmt.h"

#include <sstream>
#include <iomanip>

#include <sys/types.h>
#include <sys/stat.h>
#include <omp.h>

using namespace hpctoolkit;

SparseDB::SparseDB(const stdshim::filesystem::path& p) : dir(p), ctxMaxId(0), outputCnt(0) {
  if(dir.empty())
    util::log::fatal{} << "SparseDB doesn't allow for dry runs!";
  else
    stdshim::filesystem::create_directory(dir);
}

SparseDB::SparseDB(stdshim::filesystem::path&& p) : dir(std::move(p)), ctxMaxId(0), outputCnt(0) {
  if(dir.empty())
    util::log::fatal{} << "SparseDB doesn't allow for dry runs!";
  else
    stdshim::filesystem::create_directory(dir);
}

void SparseDB::notifyWavefront(DataClass::singleton_t ds) noexcept {
  if(((DataClass)ds).hasContexts())
    contextPrep.call_nowait([this]{ prepContexts(); });
}

void SparseDB::prepContexts() noexcept {
  std::map<unsigned int, std::reference_wrapper<const Context>> cs;
  std::function<void(const Context&)> ctx = [&](const Context& c) {
    auto id = c.userdata[src.identifier()];
    ctxMaxId = std::max(ctxMaxId, id);
    if(!cs.emplace(id, c).second)
      util::log::fatal() << "Duplicate Context identifier "
                         << c.userdata[src.identifier()] << "!";
    for(const Context& cc: c.children().iterate()) ctx(cc);
  };
  ctx(src.contexts());

  contexts.reserve(cs.size());
  for(const auto& ic: cs) contexts.emplace_back(ic.second);
}

void SparseDB::notifyThreadFinal(const Thread::Temporary& tt) {
  const auto& t = tt.thread;

  // Make sure the Context list is ready to go
  contextPrep.call([this]{ prepContexts(); });

  // Prep a quick-access Metric list, so we know what to ping.
  // TODO: Do this better with the Statistics update.
  std::vector<std::reference_wrapper<const Metric>> metrics;
  for(const Metric& m: src.metrics().iterate()) metrics.emplace_back(m);

  // Allocate the blobs needed for the final output
  std::vector<hpcrun_metricVal_t> values;
  std::vector<uint16_t> mids;
  std::vector<uint64_t> moffsets;
  std::vector<uint64_t> coffsets;
  coffsets.reserve((ctxMaxId+1)*2 + 1);  // To match up with EXML ids.

  // Now stitch together each Context's results
  for(const Context& c: contexts) {
    auto id = c.userdata[src.identifier()]*2 + 1;  // Convert to EXML id
    coffsets.resize(id+1, mids.size());
    for(const Metric& m: metrics) {
      const auto& ids = m.userdata[src.identifier()];
      auto vv = m.getFor(tt, c);
      hpcrun_metricVal_t v;
      if(vv.first != 0) {
        v.r = vv.first;
        mids.push_back(ids.first);
        moffsets.push_back(values.size());
        values.push_back(v);
      }
      if(vv.second != 0) {
        v.r = vv.second;
        mids.push_back(ids.second);
        moffsets.push_back(values.size());
        values.push_back(v);
      }
    }
  }
  coffsets.push_back(mids.size());  // One extra for ranges

  // Put together the sparse_metrics structure
  hpcrun_fmt_sparse_metrics_t sm;
  sm.tid = t.attributes.has_threadid() ? t.attributes.threadid() : 0;
  sm.num_vals = values.size();
  sm.num_cct = coffsets.size()-1;
  sm.values = values.data();
  sm.mid = mids.data();
  sm.m_offset = moffsets.data();
  sm.cct_offsets = coffsets.data();

  // Set up the output temporary file.
  stdshim::filesystem::path outfile;
  {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::ostringstream ss;
    ss << "tmp-" << world_rank << "."
       << outputCnt.fetch_add(1, std::memory_order_relaxed) << ".sparse-db";
    outfile = dir / ss.str();
  }
  std::FILE* of = std::fopen(outfile.c_str(), "wb");
  if(!of) util::log::fatal() << "Unable to open temporary sparse-db file for output!";

  // Spit it all out, and close up.
  if(hpcrun_fmt_sparse_metrics_fwrite(&sm, of) != HPCFMT_OK)
    util::log::fatal() << "Error writing out temporary sparse-db!";  
  std::fclose(of);

  // Log the output for posterity
  outputs.emplace(&t, std::move(outfile));
}

void SparseDB::write() {};

void SparseDB::merge(int threads) {
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if(world_rank != 0) {
    // Create a vector, (thread id:profile_size) 
    // TODO: need to be more unique id to identify each profile
    std::vector<std::pair<uint32_t, uint64_t>> profile_sizes;
    uint64_t my_size = 0;
    for(const auto& tp: outputs.citerate()) {
      struct stat buf;
      stat(tp.second.string().c_str(),&buf);
      my_size += buf.st_size;
      profile_sizes.emplace_back(tp.first->attributes.threadid(),buf.st_size);
    }

    //local exscan to get local offsets
    omp_set_num_threads(threads/world_size);
    exscan(profile_sizes); //now profile_sizes have corresponding local offsets 
    std::cout << "rank " << world_rank << " with size " << my_size << "\n";

    //tell rank 0 about my_size
    MPI_Gather(&my_size,1, mpi_data<uint64_t>::type,NULL,1,mpi_data<uint64_t>::type,0,MPI_COMM_WORLD);


    




    // Tell rank 0 all about our data
    {
      Gather<uint32_t> g;
      GatherStrings gs;
      printf("For outputs not zeor, size is : %d\n", outputs.size());
      for(const auto& tp: outputs.citerate()) {
        auto& attr = tp.first->attributes;
        g.add(attr.has_hostid() ? attr.hostid() : 0);
        g.add(attr.has_mpirank() ? attr.mpirank() : 0);
        g.add(attr.has_threadid() ? attr.threadid() : 0);
        g.add(attr.has_procid() ? attr.procid() : 0);
        gs.add(tp.second.string());
        printf("I am rank %d, I have tid: %d, r: %d\n", world_rank,attr.has_threadid() ? attr.threadid() : 0,attr.has_mpirank() ? attr.mpirank() : 0);
      }
      g.gatherN(6);
      gs.gatherN(7);
    }

    // Open up the output file. We use MPI's I/O substrate to make sure things
    // work in the end. Probably.
    MPI_File of;
    MPI_File_open(MPI_COMM_WORLD, (dir / "sparse.db").c_str(),
                  MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &of);

    // Make sure the file is truncated before we start writing stuff
    MPI_File_set_size(of, 0);

    // Shift into the worker code
    mergeN(threads, of);

    // Close up
    MPI_File_close(&of);
  } else {
    // Create a vector, (thread id:profile_size) 
    // TODO: need to be more unique id to identify each profile
    std::vector<std::pair<uint32_t, uint64_t>> profile_sizes;
    uint64_t my_size = 0;
    for(const auto& tp: outputs.citerate()) {
      struct stat buf;
      stat(tp.second.string().c_str(),&buf);
      my_size += buf.st_size;
      profile_sizes.emplace_back(tp.first->attributes.threadid(),buf.st_size);
    }
    //local exscan to get local offsets
    omp_set_num_threads(threads/world_size);
    exscan(profile_sizes); //now profile_sizes have corresponding local offsets 
    std::cout << "(0)rank " << world_rank << " with size " << my_size << "\n";

    std::vector<uint64_t> rank_sizes (world_size);
    //gather sizes from all workers
    MPI_Gather(&my_size,1, mpi_data<uint64_t>::type,rank_sizes.data(),1,mpi_data<uint64_t>::type,0,MPI_COMM_WORLD);
    for(auto rs: rank_sizes) std::cout << rs << " ";
    std::cout << " \n";













    // Gather the data from all the workers, build a big attributes table
    std::vector<std::pair<ThreadAttributes, stdshim::filesystem::path>> woutputs;
    {
      auto g = Gather<uint32_t>::gather0(6);
      auto gs = GatherStrings::gather0(7);
      for(std::size_t peer = 1; peer < gs.size(); peer++) {
        auto& a = g[peer];
        auto& s = gs[peer];
        for(std::size_t i = 0; i < s.size(); i++) {
          ThreadAttributes attr;
          attr.hostid(a.at(i*4));
          attr.mpirank(a.at(i*4 + 1));
          attr.threadid(a.at(i*4 + 2));
          attr.procid(a.at(i*4 + 3));
          woutputs.emplace_back(std::move(attr), std::move(s.at(i)));
        }
      }
    }

    // Copy our bits in too.
    printf("For outputs, size is : %d\n", outputs.size());
    for(const auto& tp: outputs.citerate()){
      auto& attr = tp.first->attributes;
       printf("I am rank %d (should be 0), I have tid: %d, r: %d\n", world_rank,attr.has_threadid() ? attr.threadid() : 0,attr.has_mpirank() ? attr.mpirank() : 0);
       woutputs.emplace_back(tp.first->attributes, tp.second);
    }

    // Open up the output file. We use MPI's I/O substrate to make sure things
    // work in the end. Probably.
    MPI_File of;
    MPI_File_open(MPI_COMM_WORLD, (dir / "sparse.db").c_str(),
                  MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &of);

    // Make sure the file is truncated before we start writing stuff
    MPI_File_set_size(of, 0);

    // Shift into the worker code
    merge0(threads, of, woutputs);

    // Close up and clean up
    MPI_File_close(&of);
    /* for(const auto& tp: woutputs) */
    /*   stdshim::filesystem::remove(tp.second); */
  }
}

void SparseDB::exscan(std::vector<std::pair<uint32_t, uint64_t>>& data) {
  int n = data.size();
  std::vector<uint64_t> result (n);

  #pragma omp parallel shared(result,data,n)
  {
    int p = omp_get_num_threads();
    int block = n/(p+1);

    int tid = omp_get_thread_num();
    int sum = 0;
    for(int i = tid*block; i<(tid+1)*block;i++){
      result.at(i) = data.at(i).second + sum;
      sum = result.at(i);
    }

    #pragma omp barrier   
    #pragma omp master 
    { 
      int offset = 0;
      for(int i = 1; i <= p; i++){
        offset += result.at(i*block - 1);
        result.at(i*block) = offset; 
      }
    }

    #pragma omp barrier
    int my_offset = result.at((tid+1)*block);
    for(int i = (tid+1)*block; i <= (tid+2)*block - 1; i++){
      result.at(i) = data.at(i).second + my_offset;
      my_offset = result.at(i);
    }

    #pragma omp barrier
    if(n > (p+1)*block ){
      #pragma omp master
      {
        my_offset = result.at((p+1)*block - 1);
        for(int i = (p+1)*block; i<n; i++){
          result.at(i) = data.at(i).second + my_offset;
          my_offset = result.at(i);
        }
      }    
    }

    #pragma omp barrier
    data.at(0).second = 0;
    for(int i = 1; i<n;i++){
      data.at(i).second = result.at(i-1);
    }
  }

}


