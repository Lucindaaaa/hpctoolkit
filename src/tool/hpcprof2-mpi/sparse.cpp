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
  std::vector<uint32_t> cids;
  std::vector<uint64_t> coffsets;
  coffsets.reserve(1 + (ctxMaxId+1)*2 + 1);  // To match up with EXML ids.

  // Now stitch together each Context's results
  for(const Context& c: contexts) {
    bool any = false;
    std::size_t offset = values.size();
    for(const Metric& m: metrics) {
      const auto& ids = m.userdata[src.identifier()];
      auto vv = m.getFor(tt, c);
      hpcrun_metricVal_t v;
      if(vv.first != 0) {
        v.r = vv.first;
        any = true;
        mids.push_back(ids.first);
        values.push_back(v);
      }
      if(vv.second != 0) {
        v.r = vv.second;
        any = true;
        mids.push_back(ids.second);
        values.push_back(v);
      }
    }
    if(any) {
      cids.push_back(c.userdata[src.identifier()]*2 + 1);  // Convert to EXML id
      coffsets.push_back(offset);
    }
  }

  // Put together the sparse_metrics structure
  hpcrun_fmt_sparse_metrics_t sm;
  sm.tid = t.attributes.has_threadid() ? t.attributes.threadid() : 0;
  sm.num_vals = values.size();
  sm.num_cct = contexts.size();
  sm.num_nz_cct = coffsets.size();
  sm.values = values.data();
  sm.mid = mids.data();
  sm.cct_id = cids.data();
  sm.cct_off = coffsets.data();

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

//***************************************************************************
// thread_major_sparse.db  - YUMENG
//***************************************************************************
uint64_t SparseDB::getProfileSizes(std::vector<std::pair<const hpctoolkit::Thread*, uint64_t>>& profile_sizes){
  uint64_t my_size = 0;
  for(const auto& tp: outputs.citerate()) {
    struct stat buf;
    stat(tp.second.string().c_str(),&buf);
    my_size += buf.st_size;
    profile_sizes.emplace_back(tp.first,buf.st_size);
  }
  return my_size;
}

uint32_t SparseDB::getTotalNumProfiles(uint32_t my_num_prof){
  uint32_t total_num_prof;
  MPI_Allreduce(&my_num_prof, &total_num_prof, 1, mpi_data<uint32_t>::type, MPI_SUM, MPI_COMM_WORLD);
  return total_num_prof;
}

uint64_t SparseDB::getMyOffset(uint64_t my_size,int rank){
    uint64_t my_offset;
    MPI_Exscan(&my_size, &my_offset, 1, mpi_data<uint64_t>::type, MPI_SUM, MPI_COMM_WORLD);
    if(rank == 0) my_offset = 0;
    return my_offset;
}

void SparseDB::getMyProfOffset(std::vector<std::pair<uint32_t, uint64_t>>& prof_offsets,
    std::vector<std::pair<const hpctoolkit::Thread*, uint64_t>>& profile_sizes,
    uint32_t total_prof, uint64_t my_offset, int threads)
{
  std::vector<uint64_t> tmp (profile_sizes.size());
  #pragma omp parallel for num_threads(threads)
  for(int i = 0; i<tmp.size();i++){
    tmp[i] = profile_sizes[i].second;
  }

  exscan(tmp,threads);

  #pragma omp parallel for num_threads(threads)
  for(int i = 0; i<tmp.size();i++){
    prof_offsets[i].first = profile_sizes[i].first->attributes.threadid();
    prof_offsets[i].second = tmp[i] + my_offset + (total_prof*8) + 4; //4 bytes for number of threads/profile, 8 bytes each for each offset
  }
}

void SparseDB::writeAsByte4(uint32_t val, MPI_File fh, MPI_Offset off){
  int shift = 0, num_writes = 0;
  char input[4];

  for (shift = 24; shift >= 0; shift -= 8) {
    input[num_writes] = (val >> shift) & 0xff;
    num_writes++;
  }

  MPI_Status stat;
  MPI_File_write_at(fh,off,&input,4,MPI_BYTE,&stat);

}

void SparseDB::writeAsByte8(uint64_t val, MPI_File fh, MPI_Offset off){
  int shift = 0, num_writes = 0;
  char input[8];

  for (shift = 56; shift >= 0; shift -= 8) {
    input[num_writes] = (val >> shift) & 0xff;
    num_writes++;
  }

  MPI_Status stat;
  MPI_File_write_at(fh,off,&input,8,MPI_BYTE,&stat);

}

void SparseDB::writeProfOffset(std::vector<std::pair<uint32_t, uint64_t>>& prof_offsets, MPI_File fh,
    uint32_t total_prof, int rank, int threads){
  if(rank == 0) writeAsByte4(total_prof,fh,0);

  #pragma omp parallel for num_threads(threads)
  for(int i = 0; i < prof_offsets.size(); i++) {
    int off = 4 + (prof_offsets[i].first*8);
    writeAsByte8(prof_offsets[i].second,fh,off);
  }
}

void SparseDB::writeProfiles(std::vector<std::pair<uint32_t, uint64_t>>& prof_offsets,
    std::vector<std::pair<const hpctoolkit::Thread*, uint64_t>>& profile_sizes, MPI_File fh,
    int threads){

  #pragma omp parallel for num_threads(threads)
  for(auto i = 0; i<profile_sizes.size();i++){
    //to read and write: get file name, size, offset
    const hpctoolkit::Thread* threadp = profile_sizes.at(i).first;
    uint32_t tid = (uint32_t)threadp->attributes.threadid();

    std::string fn = outputs.at(threadp).string();
    uint64_t my_prof_size = profile_sizes.at(i).second;
    MPI_Offset my_prof_offset = prof_offsets.at(i).second;
    if(tid != prof_offsets.at(i).first) std::cout << "Error in prof_offsets or profile_sizes\n";

    //get all bytes from a profile
    std::ifstream input(fn.c_str(), std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));
    input.close();

    //write at specific place
    MPI_Status stat;
    #pragma omp critical
    MPI_File_write_at(fh,my_prof_offset, bytes.data(), bytes.size(), MPI_BYTE, &stat);
  }

}

void SparseDB::writeThreadMajor(int threads, int world_rank, int world_size){
  //
  // profile_sizes: vector of (thread attributes: its own size)
  // prof_offsets: vector of (thread id: final global offset)
  // my_size: the size of this rank's profiles total
  // total_prof: total number of profiles across ranks
  //

  std::vector<std::pair<const hpctoolkit::Thread*, uint64_t>> profile_sizes;
  uint64_t my_size = getProfileSizes(profile_sizes);

  std::vector<std::pair<uint32_t, uint64_t>> prof_offsets (profile_sizes.size());
  uint32_t total_prof = getTotalNumProfiles(profile_sizes.size());
  uint64_t my_off = getMyOffset(my_size,world_rank);
  getMyProfOffset(prof_offsets,profile_sizes,total_prof, my_off, threads/world_size);

  MPI_File thread_major_f;
  MPI_File_open(MPI_COMM_WORLD, (dir / "thread_major_sparse.db").c_str(),
                  MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &thread_major_f);

  writeProfOffset(prof_offsets,thread_major_f,total_prof, world_rank,threads/world_size);
  writeProfiles(prof_offsets, profile_sizes, thread_major_f, threads/world_size);

  MPI_File_close(&thread_major_f);

}

//***************************************************************************
// cct_major_sparse.db  - YUMENG
//***************************************************************************
void vSum ( uint64_t *, uint64_t *, int *, MPI_Datatype * );

void vSum(uint64_t *invec, uint64_t *inoutvec, int *len, MPI_Datatype *dtype)
{
    int i;
    for ( i=0; i<*len; i++ )
        inoutvec[i] += invec[i];
}

void SparseDB::getCctOffset(std::vector<std::pair<uint32_t, uint64_t>>& cct_sizes,
    std::vector<std::pair<uint32_t, uint64_t>>& cct_off,int threads)
{
  std::vector<uint64_t> tmp (cct_sizes.size());
  #pragma omp parallel for num_threads(threads)
  for(int i = 0; i<tmp.size();i++){
    tmp[i] = cct_sizes[i].second;
  }

  exscan(tmp,threads); //get local offsets

  //sum up local offsets to get global offsets
  MPI_Op vectorSum;
  MPI_Op_create((MPI_User_function *)vSum, 1, &vectorSum);
  MPI_Allreduce(MPI_IN_PLACE, tmp.data(), tmp.size() ,mpi_data<uint64_t>::type, vectorSum, MPI_COMM_WORLD);

  #pragma omp parallel for num_threads(threads)
  for(int i = 0; i<tmp.size();i++){
    cct_off[i].first = cct_sizes[i].first;
    cct_off[i].second = tmp[i];
  }

  MPI_Op_free(&vectorSum);

}

void SparseDB::getMyCCTs(std::vector<std::pair<uint32_t, uint64_t>>& cct_off,
    std::vector<uint32_t>& my_ccts,uint64_t last_cct_size, int num_ranks, int rank)
{
  MPI_Allreduce(MPI_IN_PLACE, &last_cct_size, 1 ,mpi_data<uint64_t>::type, MPI_SUM, MPI_COMM_WORLD);

  uint64_t total_size = cct_off[cct_off.size()-1].second + last_cct_size;
  uint64_t max_size_per_rank = round(total_size/num_ranks);
  uint64_t my_start = rank*max_size_per_rank;
  uint64_t my_end = (rank == num_ranks-1) ? total_size : (rank+1)*max_size_per_rank;

  if(rank == 0) my_ccts.emplace_back(0);
  for(int i = 2; i<cct_off.size(); i++){
    if(cct_off[i].second>my_start && cct_off[i].second <= my_end) my_ccts.emplace_back(cct_off[i-1].first);
  }
  if(rank == num_ranks-1) my_ccts.emplace_back(cct_off[cct_off.size()-1].first);


}



//***************************************************************************
// general - YUMENG
//***************************************************************************

void SparseDB::merge(int threads, std::size_t ctxcnt) {
  int world_rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  {
    util::log::debug msg{false};  // Switch to true for CTX id printouts
    msg << "CTXs (" << world_rank << ":" << outputs.size() << "): "
        << ctxcnt;
  }

  writeThreadMajor(threads,world_rank,world_size);

/* TEMP: test some cct major functions
  std::vector<std::pair<uint32_t, uint64_t>> cct_sizes (30);
  for(int i =0;i<30;i++){
    cct_sizes[i].first = i;
    cct_sizes[i].second = i*10;
  }

  std::vector<std::pair<uint32_t, uint64_t>> cct_off (cct_sizes.size());
  std::vector<uint32_t> my_cct;
  getCctOffset(cct_sizes,cct_off,threads/world_size);
  for(auto c :cct_off) std::cout << " " << c.second;
  std::cout << "\n\n";
  getMyCCTs(cct_off,my_cct,cct_sizes[cct_sizes.size()-1].second,world_size, world_rank);
  std::cout << "Rank " << world_rank << ": ";
  for(auto c :my_cct) std::cout << " " << c;
  std::cout << "\n\n";

  std::cout << "size I am responsible for : " << (cct_off[my_cct[my_cct.size()-1]].second - cct_off[my_cct[0]].second + cct_sizes[my_cct[my_cct.size()-1]].second) <<"\n";
*/
}


#if 0
//Jonathon's original code
void SparseDB::merge(int threads) {
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  if(world_rank != 0) {
    // Tell rank 0 all about our data
    {
      Gather<uint32_t> g;
      GatherStrings gs;
      for(const auto& tp: outputs.citerate()) {
        auto& attr = tp.first->attributes;
        g.add(attr.has_hostid() ? attr.hostid() : 0);
        g.add(attr.has_mpirank() ? attr.mpirank() : 0);
        g.add(attr.has_threadid() ? attr.threadid() : 0);
        g.add(attr.has_procid() ? attr.procid() : 0);
        gs.add(tp.second.string());
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
    for(const auto& tp: outputs.citerate()){
      auto& attr = tp.first->attributes;
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
#endif

void SparseDB::exscan(std::vector<uint64_t>& data, int threads) {
  int n = data.size();
  int rounds = ceil(std::log2(n));
  std::vector<uint64_t> tmp (n);

  for(int i = 0; i<rounds; i++){
    #pragma omp parallel for num_threads(threads)
    for(int j = 0; j < n; j++){
      int p = (int)pow(2.0,i);
      tmp.at(j) = (j<p) ?  data.at(j) : data.at(j)+data.at(j-p);
    }
    if(i<rounds-1) data = tmp;
  }

  if(n>0) data[0] = 0;
  #pragma omp parallel for num_threads(threads)
  for(int i = 1; i < n; i++){
    data[i] = tmp[i-1];
  }
}
