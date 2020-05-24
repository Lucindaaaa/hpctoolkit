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
//
/*EXAMPLE
[Number of profiles] (*not printed in hpcproftt)
[Profile informations (thread id : number of nonzero values : number of nonzero CCTs : offset)
  (0:186:112:65258   1:136:74:98930   2:138:75:107934   3:136:74:6224   4:131:71:70016   5:148:85:91202   ...)
]
[
  (values:  4.02057  4.02057  4.02057  3.98029  0.01816  0.00154  ...)
  (metric id: 1 1 1 1 1 0 ...)
  (cct offsets (cct id : offset): 1:0 7:1 9:2 21:3 23:4 25:5 ...)
]
...same [sparse metrics] for all rest threads 
*/
//
// SIZE CHART: data(size in bytes)
// Number of profiles (4)
// [Profile informations] 
//    thread id (4) : number of nonzero values (8) : number of nonzero CCTs (4) : offset(8)
// [sparse metrics] 
//    non-zero values (8)
//    Metric IDs of non-zero values (2)
//    cct id (4) : offset (8)

//***************************************************************************
uint64_t SparseDB::getProfileSizes(std::vector<std::pair<const hpctoolkit::Thread*, uint64_t>>& profile_sizes){
  uint64_t my_size = 0;
  for(const auto& tp: outputs.citerate()) {
    struct stat buf;
    stat(tp.second.string().c_str(),&buf);
    my_size += (buf.st_size - TMS_prof_skip_SIZE);
    profile_sizes.emplace_back(tp.first,(buf.st_size - TMS_prof_skip_SIZE));    
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

  exscan<uint64_t>(tmp,threads);

  #pragma omp parallel for num_threads(threads)
  for(int i = 0; i<tmp.size();i++){
    prof_offsets[i].first = profile_sizes[i].first->attributes.threadid();
    prof_offsets[i].second = tmp[i] + my_offset + (total_prof * TMS_prof_info_SIZE) + TMS_total_prof_SIZE; 
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


void SparseDB::writeAsByteX(std::vector<char> val, size_t size, MPI_File fh, MPI_Offset off){
  MPI_Status stat;
  MPI_File_write_at(fh,off,val.data(),size,MPI_BYTE,&stat);
}

void SparseDB::writeProfInfo(std::vector<std::pair<uint32_t, uint64_t>>& prof_offsets, std::unordered_map<uint32_t,std::vector<char>>& prof_infos,
    MPI_File fh, uint32_t total_prof, int rank, int threads){

  if(rank == 0) writeAsByte4(total_prof,fh,0);

  #pragma omp parallel for num_threads(threads)
  for(int i = 0; i < prof_offsets.size(); i++) {
    MPI_Offset off = TMS_total_prof_SIZE + (prof_offsets[i].first*TMS_prof_info_SIZE);
    std::vector<char> info = prof_infos[prof_offsets[i].first]; 
    writeAsByteX(info,TMS_prof_skip_SIZE,fh,off);
    writeAsByte8(prof_offsets[i].second,fh,off+TMS_prof_skip_SIZE);
  }
}

void SparseDB::collectCctMajorData(uint64_t* cct_local_sizes, std::vector<std::set<uint16_t>>& cct_nzmids, std::vector<char>& bytes)
{
  uint64_t num_val;
  uint32_t num_nzcct;
  interpretByte8(&num_val, bytes.data()+TMS_tid_SIZE);
  interpretByte4(&num_nzcct, bytes.data()+ TMS_tid_SIZE + TMS_num_val_SIZE);
  int before_cct_SIZE = TMS_prof_skip_SIZE + num_val * (TMS_val_SIZE + TMS_mid_SIZE);
  int before_mid_SIZE = TMS_prof_skip_SIZE + num_val * TMS_val_SIZE;

  for(int i = 0; i < num_nzcct; i++){
    uint32_t cct_id;

    //local sizes
    uint64_t cct_offset;
    uint64_t next_cct_offset = num_val;
    interpretByte4(&cct_id, bytes.data()+ before_cct_SIZE + TMS_cct_pair_SIZE*i);
    interpretByte8(&cct_offset, bytes.data()+ before_cct_SIZE + TMS_cct_pair_SIZE*i + TMS_cct_id_SIZE);
    if(i<num_nzcct-1) interpretByte8(&next_cct_offset, bytes.data()+ before_cct_SIZE + TMS_cct_pair_SIZE*(i+1) + TMS_cct_id_SIZE);
    uint64_t num_val_this_cct = next_cct_offset - cct_offset;
    cct_local_sizes[CCTLOCALSIZESIDX(cct_id)] += num_val_this_cct;    

    
    //nz_mids (number of non-zero values = number of non-zero metric ids)
    for(int m = 0; m < num_val_this_cct; m++){
      uint16_t mid;
      interpretByte2(&mid, bytes.data()+ before_mid_SIZE + (cct_offset+m) * TMS_mid_SIZE);
      if(cct_nzmids[CCTLOCALSIZESIDX(cct_id)].size() == 0){
        std::set<uint16_t> mids;
        cct_nzmids[CCTLOCALSIZESIDX(cct_id)] = mids;
      }
      cct_nzmids[CCTLOCALSIZESIDX(cct_id)].insert(mid);
    }
    
  }

}


void SparseDB::writeProfiles(std::vector<std::pair<uint32_t, uint64_t>>& prof_offsets, std::vector<uint64_t>& cct_local_sizes,
    std::vector<std::set<uint16_t>>& cct_nzmids,
    std::vector<std::pair<const hpctoolkit::Thread*, uint64_t>>& profile_sizes, std::unordered_map<uint32_t,std::vector<char>>& prof_infos, 
    MPI_File fh, int threads){

  uint64_t cct_local_sizes_arr[cct_local_sizes.size()] = {0};

  #pragma omp parallel num_threads(threads)
  {
    std::set<uint16_t> empty;
    std::vector<std::set<uint16_t>> thread_cct_nzmids (cct_nzmids.size(),empty);

    #pragma omp for reduction(+:cct_local_sizes_arr[:cct_local_sizes.size()]) 
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


      //extract bytes for profile information
      std::vector<char> info (TMS_prof_skip_SIZE);
      std::copy(bytes.data(),bytes.data()+TMS_prof_skip_SIZE,info.begin());
      //prof_infos.emplace(tid,info);
      prof_infos[tid] = info;

      //collect cct local sizes
      collectCctMajorData(cct_local_sizes_arr,thread_cct_nzmids,bytes);

      //write at specific place
      MPI_Status stat;
      MPI_File_write_at(fh,my_prof_offset, bytes.data()+TMS_prof_skip_SIZE, bytes.size()-TMS_prof_skip_SIZE, MPI_BYTE, &stat);
    }

    #pragma omp critical
    {
      for(int j = 0; j<cct_nzmids.size(); j++){
        //std::set<uint16_t> union_nzmids;
        std::set_union(cct_nzmids[j].begin(), cct_nzmids[j].end(),
              thread_cct_nzmids[j].begin(), thread_cct_nzmids[j].end(),
              std::inserter(cct_nzmids[j], cct_nzmids[j].begin()));
        //cct_nzmids[j] = union_nzmids;
      }
    }
  }
  

  std::copy(cct_local_sizes_arr,cct_local_sizes_arr+cct_local_sizes.size(),cct_local_sizes.begin());
  

}

void SparseDB::writeThreadMajor(int threads, int world_rank, int world_size, std::vector<uint64_t>& cct_local_sizes,std::vector<std::set<uint16_t>>& cct_nzmids){
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


  std::unordered_map<uint32_t,std::vector<char>> prof_infos(profile_sizes.size());
  writeProfiles(prof_offsets, cct_local_sizes, cct_nzmids, profile_sizes, prof_infos, thread_major_f, threads/world_size);
  writeProfInfo(prof_offsets,prof_infos, thread_major_f,total_prof, world_rank,threads/world_size);

  MPI_File_close(&thread_major_f);

}

//***************************************************************************
// cct_major_sparse.db  - YUMENG
//
/*EXAMPLE
[Number of CCTs] (*not printed in hpcproftt)
[CCT informations (cct id : number of nonzero values : number of nonzero metric ids : offset)
  (0:186:112:65258   1:136:74:98930   2:138:75:107934   3:136:74:6224   4:131:71:70016   5:148:85:91202   ...)
]
[
  (values:  4.02057  4.02057  4.02057  3.98029  0.01816  0.00154  ...)
  (thread id: 1 1 1 1 1 0 ...)
  (metric_id offsets (metric id : offset): 1:0 7:1 9:2 21:3 23:4 25:5 ...)
]
...same [sparse metrics] for all rest ccts 
*/
//
// SIZE CHART: data(size in bytes)
// Number of ccts (4)
// [CCT informations] 
//    cct id (4) : number of nonzero values (8) : number of nonzero metric ids (2) : offset(8)
// [sparse metrics] 
//    non-zero values (8)
//    thread IDs of non-zero values (4)
//    metric id (2) : offset (8)
//***************************************************************************
void vSum ( uint64_t *, uint64_t *, int *, MPI_Datatype * );

void vSum(uint64_t *invec, uint64_t *inoutvec, int *len, MPI_Datatype *dtype)
{
    int i;
    for ( i=0; i<*len; i++ )
        inoutvec[i] += invec[i];
}

void SparseDB::unionMids(std::vector<std::set<uint16_t>>& cct_nzmids, int rank, int num_proc)
{
  //convert to a long vector with stopper
  uint16_t stopper = -1;
  std::vector<uint16_t> rank_all_mids;
  for(auto cct : cct_nzmids){
    for(auto mid: cct){
      rank_all_mids.emplace_back(mid);
    }
    rank_all_mids.emplace_back(stopper);
  }

  //prepare for later gatherv
  int rank_mids_size = rank_all_mids.size();
  int *mids_sizes = NULL;
  if(rank == 0) mids_sizes = (int *) malloc(num_proc * sizeof(int));
  MPI_Gather(&rank_mids_size, 1, MPI_INT, mids_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> mids_disps (num_proc);
  int total_size = 0;
  uint16_t *global_all_mids = NULL;
  if(rank == 0){
    for(int i = 0; i<num_proc; i++) mids_disps[i] = mids_sizes[i];
    exscan<int>(mids_disps,1); //TODO: temporarily use 1 thread, try add omp 
    total_size = mids_disps[num_proc-1] + mids_sizes[num_proc-1];
    global_all_mids = (uint16_t *) malloc(total_size * sizeof(uint16_t));
  }

  //gather all the rank_all_mids (i.e. cct_nzmids) to root
  MPI_Gatherv(rank_all_mids.data(),rank_mids_size, mpi_data<uint16_t>::type, \
    global_all_mids, mids_sizes, mids_disps.data(), mpi_data<uint16_t>::type, 0, MPI_COMM_WORLD);

  if(rank == 0){
    int num_stopper = 0;
    int num_cct     = cct_nzmids.size();
    for(int i = 0; i< total_size; i++) {
      uint16_t mid = global_all_mids[i];
      if(mid == stopper){
        num_stopper++;
      }else{
        cct_nzmids[num_stopper % num_cct].insert(mid); 
        //printf("cct %d, insert %d\n", num_stopper % num_cct, mid);
      }
    }
  }


}

//input: local sizes for all cct, output: final offsets for all cct
void SparseDB::getCctOffset(std::vector<uint64_t>& cct_sizes, std::vector<std::set<uint16_t>> cct_nzmids,
    std::vector<std::pair<uint32_t, uint64_t>>& cct_off,int threads, int rank)
{
  std::vector<uint64_t> tmp (cct_sizes.size());
  #pragma omp parallel for num_threads(threads)
  for(int i = 0; i<tmp.size();i++){
    tmp[i] = cct_sizes[i] * CMS_val_tid_pair_SIZE;
    if(rank == 0) tmp[i] += cct_nzmids[i].size() * CMS_m_pair_SIZE; 
  }

  exscan<uint64_t>(tmp,threads); //get local offsets

  //sum up local offsets to get global offsets
  MPI_Op vectorSum;
  MPI_Op_create((MPI_User_function *)vSum, 1, &vectorSum);
  MPI_Allreduce(MPI_IN_PLACE, tmp.data(), tmp.size() ,mpi_data<uint64_t>::type, vectorSum, MPI_COMM_WORLD);

  #pragma omp parallel for num_threads(threads)
  for(int i = 0; i<tmp.size();i++){
    cct_off[i].first = CCTID(i);
    cct_off[i].second = tmp[i];
  }

  MPI_Op_free(&vectorSum);

}

//input: final offsets for all cct, output: my(rank) own responsible cct list
void SparseDB::getMyCCTs(std::vector<std::pair<uint32_t, uint64_t>>& cct_off,
    std::vector<uint32_t>& my_ccts,uint64_t& last_cct_size, uint64_t& total_size, int num_ranks, int rank)
{
  MPI_Allreduce(MPI_IN_PLACE, &last_cct_size, 1 ,mpi_data<uint64_t>::type, MPI_SUM, MPI_COMM_WORLD);

  total_size = cct_off[cct_off.size()-1].second + last_cct_size;
  uint64_t max_size_per_rank = round(total_size/num_ranks);
  uint64_t my_start = rank*max_size_per_rank;
  uint64_t my_end = (rank == num_ranks-1) ? total_size : (rank+1)*max_size_per_rank;

  if(rank == 0) my_ccts.emplace_back(FIRST_CCT_ID); 
  for(int i = 2; i<cct_off.size(); i++){
    if(cct_off[i].second>my_start && cct_off[i].second <= my_end) my_ccts.emplace_back(cct_off[i-1].first);
  }
  if(rank == num_ranks-1) my_ccts.emplace_back(cct_off[cct_off.size()-1].first);


}

void SparseDB::updateCctOffset(std::vector<std::pair<uint32_t, uint64_t>>& cct_off,uint64_t& total_size, size_t ctxcnt, int threads)
{
  #pragma omp parallel for num_threads(threads)
  for(int i =0; i<ctxcnt; i++){
    cct_off[i].second += ctxcnt * CMS_cct_info_SIZE + CMS_num_cct_SIZE;
  }
  total_size += ctxcnt * CMS_cct_info_SIZE + CMS_num_cct_SIZE;
}

void SparseDB::readAsByte4(uint32_t *val, MPI_File fh, MPI_Offset off){
  uint32_t v = 0;
  int shift = 0, num_reads = 0;
  char input[4];

  MPI_Status stat;
  MPI_File_read_at(fh,off,&input,4,MPI_BYTE,&stat);
  
  for (shift = 24; shift >= 0; shift -= 8) {
    v |= ((uint32_t)(input[num_reads] & 0xff) << shift); 
    num_reads++;
  }

  *val = v;

}

void SparseDB::readAsByte8(uint64_t *val, MPI_File fh, MPI_Offset off){
  uint32_t v = 0;
  int shift = 0, num_reads = 0;
  char input[8];

  MPI_Status stat;
  MPI_File_read_at(fh,off,&input,8,MPI_BYTE,&stat);
  
  for (shift = 56; shift >= 0; shift -= 8) {
    v |= ((uint64_t)(input[num_reads] & 0xff) << shift); 
    num_reads++;
  }

  *val = v;

}

void SparseDB::interpretByte2(uint16_t *val, char *input){
  uint16_t v = 0;
  int shift = 0, num_reads = 0;

  for (shift = 8; shift >= 0; shift -= 8) {
    v |= ((uint16_t)(input[num_reads] & 0xff) << shift); 
    num_reads++;
  }

  *val = v;
}

void SparseDB::interpretByte4(uint32_t *val, char *input){
  uint32_t v = 0;
  int shift = 0, num_reads = 0;

  for (shift = 24; shift >= 0; shift -= 8) {
    v |= ((uint32_t)(input[num_reads] & 0xff) << shift); 
    num_reads++;
  }

  *val = v;
}

void SparseDB::interpretByte8(uint64_t *val, char *input){
  uint64_t v = 0;
  int shift = 0, num_reads = 0;

  for (shift = 56; shift >= 0; shift -= 8) {
    v |= ((uint64_t)(input[num_reads] & 0xff) << shift); 
    num_reads++;
  }

  *val = v;
}

void SparseDB::readCCToffsets(std::vector<std::pair<uint32_t,uint64_t>>& cct_offsets,
    MPI_File fh,MPI_Offset off){
    
    int count = cct_offsets.size() * TMS_cct_pair_SIZE; //each cct has a 4-byte cct id and 8-byte cct offset
    char input[count];

    MPI_Status stat;
    MPI_File_read_at(fh,off,&input,count,MPI_BYTE,&stat);

    for(int i = 0; i<count; i += TMS_cct_pair_SIZE){
      uint32_t cct_id;
      uint64_t cct_off;
      interpretByte4(&cct_id, input+i);
      interpretByte8(&cct_off, input+i+TMS_cct_id_SIZE);
      auto p = std::make_pair(cct_id, cct_off);
      cct_offsets[i/12] = p;
    }
    
}

//given full cct offsets and a group of cct ids, return a vector of <cct id, offset> with number of nzcct in this group + 1
int SparseDB::binarySearchCCTid(std::vector<uint32_t>& cct_ids,
    std::vector<std::pair<uint32_t,uint64_t>>& profile_cct_offsets,
    std::vector<std::pair<uint32_t,uint64_t>>& my_cct_offsets)
{
  int n = profile_cct_offsets.size();
  int m;
  int found = 0;
  uint32_t target;

  for(int i = 0; i<cct_ids.size(); i++){
    target = cct_ids[i];
    if(found){ // if already found one, no need to binary search again
      m += 1;
      if(profile_cct_offsets[m].first == target){
        my_cct_offsets.emplace_back(target,profile_cct_offsets[m].second);
      }else if(profile_cct_offsets[m].first > target){
        //my_cct_offsets.emplace_back(target,profile_cct_offsets[m-1].second);
        m -= 1; //back to original since this might be next target
      }else{ //profile_cct_offsets[m].first < target, should not happen
        printf("ERROR: SparseDB::binarySearchCCTid(): cct id %d in a profile does not exist in the full cct list.\n", profile_cct_offsets[m].first );
        return -1;
      }
    }else{
      int L = 0;
      int R = n - 1;
      while(L<=R){
        m = (L+R)/2;
        if(profile_cct_offsets[m].first < target){
          L = m + 1;
        }else if(profile_cct_offsets[m].first > target){
          R = m - 1;
        }else{ //find match
          my_cct_offsets.emplace_back(target,profile_cct_offsets[m].second);
          found = 1;
          break;
        }
      }
      //if(!found) my_cct_offsets.emplace_back(target,0);

    }    
  }

  if(m == profile_cct_offsets.size()-1){
    my_cct_offsets.emplace_back(NEED_NUM_VAL,NEED_NUM_VAL);
  }else{
    my_cct_offsets.emplace_back(profile_cct_offsets[m+1].first,profile_cct_offsets[m+1].second);
  }
  

  return 1;
}


void SparseDB::readOneProfile(std::vector<uint32_t>& cct_ids, ProfileInfo prof_info,
    std::unordered_map<uint32_t,std::vector<DataBlock>>& cct_data_pairs,MPI_File fh)
{
  //TODO: val and mid write together => one seek per cct group

  //find the corresponding cct and its offset in values and mids
  MPI_Offset offset = prof_info.offset;
  MPI_Offset cct_offsets_offset = offset + prof_info.num_val * (TMS_val_SIZE + TMS_mid_SIZE);
  std::vector<std::pair<uint32_t,uint64_t>> full_cct_offsets (prof_info.num_nzcct);
  readCCToffsets(full_cct_offsets,fh,cct_offsets_offset);
  

  std::vector<std::pair<uint32_t,uint64_t>> my_cct_offsets;
  binarySearchCCTid(cct_ids,full_cct_offsets,my_cct_offsets);


  //read all values and metric ids for this group of cct at once
  if(my_cct_offsets[my_cct_offsets.size()-1].second == NEED_NUM_VAL) my_cct_offsets[my_cct_offsets.size()-1].second = prof_info.num_val;
  MPI_Offset val_start_pos = offset + my_cct_offsets[0].second * TMS_val_SIZE;
  int val_count = (my_cct_offsets[my_cct_offsets.size()-1].second - my_cct_offsets[0].second) * TMS_val_SIZE;
  MPI_Offset mid_start_pos = offset + prof_info.num_val * TMS_val_SIZE + my_cct_offsets[0].second * TMS_mid_SIZE;
  int mid_count = (my_cct_offsets[my_cct_offsets.size()-1].second - my_cct_offsets[0].second) * TMS_mid_SIZE;
  char vinput[val_count];
  char minput[mid_count];
  MPI_Status stat;
  if(val_count != 0) MPI_File_read_at(fh,val_start_pos,&vinput,val_count,MPI_BYTE,&stat);
  if(mid_count != 0) MPI_File_read_at(fh,mid_start_pos,&minput,mid_count,MPI_BYTE,&stat);

  //for each cct, keep track of the values,metric ids, and thread ids
  for(int c = 0; c<my_cct_offsets.size()-1; c++) 
  {
    uint32_t cct_id = my_cct_offsets[c].first;
    uint64_t num_val_this_cct = my_cct_offsets[c+1].second - my_cct_offsets[c].second;
    char* val_start_this_cct = vinput + (my_cct_offsets[c].second - my_cct_offsets[0].second) * TMS_val_SIZE;
    char* mid_start_this_cct = minput + (my_cct_offsets[c].second - my_cct_offsets[0].second) * TMS_mid_SIZE;
    for(int i = 0; i<num_val_this_cct; i++){
      //get a pair of val and mid
      hpcrun_metricVal_t val;
      interpretByte8(&val.bits,val_start_this_cct+i*TMS_val_SIZE);
      uint16_t mid;
      interpretByte2(&mid,mid_start_this_cct + i*TMS_mid_SIZE);

      //store them
      std::unordered_map<uint32_t,std::vector<DataBlock>>::iterator got = cct_data_pairs.find (cct_id);

      if ( got == cct_data_pairs.end() ){
        //this cct doesn't exist in cct_data_paris yet
        
        DataBlock data;
        data.mid = mid;
        std::vector<std::pair<hpcrun_metricVal_t,uint32_t>> values_tids;
        values_tids.emplace_back(val,prof_info.tid);
        data.values_tids = values_tids;
        std::vector<DataBlock> datas;
        datas.emplace_back(data);
        cct_data_pairs.emplace(cct_id,datas);
      }else{
        //find the DataBlock with mid
        std::vector<DataBlock> datas = got->second; 

        std::vector<DataBlock>::iterator it = std::find_if(datas.begin(), datas.end(), 
                       [&mid] (const DataBlock& d) { 
                          return d.mid == mid; 
                       });

        if(it != datas.end()){ 
          it->values_tids.emplace_back(val,prof_info.tid);
        }else{
          DataBlock data;
          data.mid = mid;
          std::vector<std::pair<hpcrun_metricVal_t,uint32_t>> values_tids;
          values_tids.emplace_back(val,prof_info.tid);
          data.values_tids = values_tids;
          datas.emplace_back(data);
        }

        got->second = datas;

      }//END of cct_id found in cct_data_pair 
    }//END of storing values and thread ids for this cct
  }//END of storing values and thread ids for ALL cct

}

void SparseDB::readProfileInfo(std::vector<ProfileInfo>& prof_info, MPI_File fh)
{
  uint32_t num_prof;
  readAsByte4(&num_prof,fh,0);
  int count = num_prof * TMS_prof_info_SIZE; 
  char input[count];

  MPI_Status stat;
  MPI_File_read_at(fh,TMS_total_prof_SIZE,&input,count,MPI_BYTE,&stat);

  for(int i = 0; i<count; i += TMS_prof_info_SIZE){
    uint32_t tid;
    uint64_t num_val;
    uint32_t num_nzcct;
    uint64_t offset;
    interpretByte4(&tid, input + i);
    interpretByte8(&num_val, input + i + TMS_tid_SIZE);
    interpretByte4(&num_nzcct, input + i + TMS_tid_SIZE + TMS_num_val_SIZE);
    interpretByte8(&offset, input + i + TMS_tid_SIZE + TMS_num_val_SIZE + TMS_num_nzcct_SIZE);
    ProfileInfo pi = {tid, num_val, num_nzcct, offset};
    prof_info.emplace_back(pi);
  }
}

void SparseDB::convertToByte8(uint64_t val, char* bytes){
  int shift = 0, num_writes = 0;
  //char input[8];

  for (shift = 56; shift >= 0; shift -= 8) {
    //input[num_writes] = (val >> shift) & 0xff;
    bytes[num_writes] = (val >> shift) & 0xff;
    num_writes++;
  }

  //*bytes = input;

}

void SparseDB::convertToByte4(uint32_t val, char* bytes){
  int shift = 0, num_writes = 0;
  //char input[4];

  for (shift = 24; shift >= 0; shift -= 8) {
    //input[num_writes] = (val >> shift) & 0xff;
    bytes[num_writes] = (val >> shift) & 0xff;
    num_writes++;
  }

  //*bytes = input;

}

void SparseDB::convertToByte2(uint16_t val, char* bytes){
  int shift = 0, num_writes = 0;
  //char input[2];

  for (shift = 8; shift >= 0; shift -= 8) {
    //input[num_writes] = (val >> shift) & 0xff;
    bytes[num_writes] = (val >> shift) & 0xff;
    num_writes++;
  }

  //*bytes = input;

}


void SparseDB::dataPairs2Bytes(std::unordered_map<uint32_t,std::vector<DataBlock>>& cct_data_pairs, 
    std::vector<std::pair<uint32_t, uint64_t>>& cct_off,std::vector<uint32_t> cct_ids,
    std::vector<char>& info_bytes,std::vector<char>& metrics_bytes)
{
  int bytecnt = 0;
  //NOTICE: info_bytes and metrics_bytes contain the data for the number of ccts in this group (cct_data_pairs)
  uint64_t first_cct_off =  cct_off[CCTLOCALSIZESIDX(cct_ids[0])].second; 
  for(int i = 0; i<cct_ids.size(); i++ ){
    //INFO_BYTES
    uint32_t cct_id = cct_ids[i];
    uint64_t num_val = 0;
    uint16_t num_nzmid = 0;
    uint64_t offset = cct_off[CCTLOCALSIZESIDX(cct_id)].second; 
    std::unordered_map<uint32_t,std::vector<DataBlock>>::iterator got = cct_data_pairs.find (cct_id);

    if(got != cct_data_pairs.end()){ //cct_data_pairs has the cct_id
      std::vector<DataBlock> datas = got->second;
      //INFO_BYTES
      num_nzmid = datas.size();
      //METRIC_BYTES
      uint64_t this_cct_start_pos = offset - first_cct_off;

      uint64_t pre_val_tid_pair_size = 0;
      for(int j =0; j<num_nzmid; j++){
        DataBlock d = datas[j];
        //INFO_BYTES
        datas[j].num_values = d.values_tids.size(); //set up for later use
        num_val += datas[j].num_values ;

        //METRIC_BYTES - val_tid_pair
        for(int k = 0; k<d.values_tids.size(); k++){
          auto pair = d.values_tids[k];
          convertToByte8(pair.first.bits, metrics_bytes.data() + this_cct_start_pos + pre_val_tid_pair_size + k * CMS_val_tid_pair_SIZE);
          bytecnt += 8;
          convertToByte4(pair.second, metrics_bytes.data() + this_cct_start_pos + pre_val_tid_pair_size + k * CMS_val_tid_pair_SIZE + CMS_val_SIZE);
          bytecnt += 4;
        }
        pre_val_tid_pair_size = num_val * CMS_val_tid_pair_SIZE;

      }//END OF DataBlock loop

      //METRIC_BYTES - metricID_offset_pair
      uint64_t m_off = 0;
      for(int j =0; j<num_nzmid; j++){   
       // printf("%d %d %d",metrics_bytes.data(), metrics_bytes.data() + this_cct_start_pos + num_val * CMS_val_tid_pair_SIZE + j * CMS_m_pair_SIZE,\
          metrics_bytes.data() + this_cct_start_pos + num_val * CMS_val_tid_pair_SIZE + j * CMS_m_pair_SIZE + CMS_mid_SIZE); 
        convertToByte2(datas[j].mid, metrics_bytes.data() + this_cct_start_pos + num_val * CMS_val_tid_pair_SIZE + j * CMS_m_pair_SIZE);
        bytecnt += 2;
        convertToByte8(m_off, metrics_bytes.data() + this_cct_start_pos + num_val * CMS_val_tid_pair_SIZE + j * CMS_m_pair_SIZE + CMS_mid_SIZE);
        bytecnt += 8;
        m_off += datas[j].num_values;
      }


    }//END OF if cct_data_pairs has the cct_id


    uint64_t pre = i * CMS_cct_info_SIZE;
    convertToByte4(cct_id,info_bytes.data() + pre);
    convertToByte8(num_val,info_bytes.data() + pre + CMS_cct_id_SIZE);
    convertToByte2(num_nzmid,info_bytes.data() + pre + CMS_cct_id_SIZE + CMS_num_val_SIZE);
    convertToByte8(offset,info_bytes.data() + pre + CMS_cct_id_SIZE + CMS_num_val_SIZE + CMS_num_nzmid_SIZE);
  } //END OF CCT LOOP

  if(bytecnt != metrics_bytes.size()) printf("cct %d DID NOT FILL ALL THE BYTES! bytecnt %d, need %d\n", cct_ids[0], bytecnt, metrics_bytes.size());
}

void SparseDB::rwOneCCTgroup(std::vector<uint32_t>& cct_ids, std::vector<ProfileInfo>& prof_info,
    std::vector<std::pair<uint32_t, uint64_t>>& cct_off, uint64_t total_size, MPI_File fh, MPI_File ofh)
{
  std::unordered_map<uint32_t,std::vector<DataBlock>> cct_data_pairs;
  //std::vector<std::pair<uint32_t,std::vector<DataBlock>>> cct_data_pairs;

  //read all profiles for this cct_ids group
  for(int i = 0; i < prof_info.size(); i++){
    ProfileInfo pi = prof_info[i];
    readOneProfile(cct_ids,pi,cct_data_pairs,fh);
  }

/*  //TEST:for(const auto& tp: outputs.citerate()) {
  for(auto cdp : cct_data_pairs){
    uint32_t cctid = cdp.first;
    std::cout << "cct id " << cctid << ":\n";
    std::vector<DataBlock> datas = cdp.second;

    for(auto d:datas){
      uint16_t mid = d.mid;
      std::cout << "  metric id " << mid << ": (val:tid) \n";
      for(auto vt:d.values_tids) printf("  (%g:%d)  ",vt.first.r, vt.second );
      std::cout << "\n";
    }

    std::cout << "\n";
    
  }
*/
  

  //TODO: for each cct id and metric id, sort the value:thread id pair and finalize num_values
  //write for this cct_ids group
  std::vector<char> info_bytes (CMS_cct_info_SIZE * cct_ids.size());
  uint32_t first_cct_id = cct_ids[0];
  uint32_t last_cct_id = cct_ids[cct_ids.size()-1];
  int metric_bytes_size = (last_cct_id == cct_off[cct_off.size()-1].first) ? total_size - cct_off[CCTLOCALSIZESIDX(first_cct_id)].second \
        : cct_off[CCTLOCALSIZESIDX(last_cct_id+2)].second - cct_off[CCTLOCALSIZESIDX(first_cct_id)].second;
  std::vector<char> metrics_bytes (metric_bytes_size);
  dataPairs2Bytes(cct_data_pairs,cct_off,cct_ids,info_bytes,metrics_bytes);

  MPI_Status stat;
  MPI_Offset info_off = CMS_num_cct_SIZE + CCTLOCALSIZESIDX(cct_ids[0]) * CMS_cct_info_SIZE;
  MPI_File_write_at(ofh,info_off,info_bytes.data(),info_bytes.size(),MPI_BYTE,&stat);

  MPI_Offset metrics_off = cct_off[CCTLOCALSIZESIDX(first_cct_id)].second;
  MPI_File_write_at(ofh,metrics_off,metrics_bytes.data(),metrics_bytes.size(),MPI_BYTE,&stat);




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

  std::vector<uint64_t> cct_local_sizes (ctxcnt,0);
  std::set<uint16_t> empty;
  std::vector<std::set<uint16_t>> cct_nzmids(ctxcnt,empty);
  writeThreadMajor(threads,world_rank,world_size, cct_local_sizes,cct_nzmids);


  unionMids(cct_nzmids,world_rank,world_size);


  std::vector<std::pair<uint32_t, uint64_t>> cct_off (ctxcnt);
  std::vector<uint32_t> my_cct;
  getCctOffset(cct_local_sizes,cct_nzmids, cct_off,threads/world_size, world_rank);
  uint64_t last_cct_size = cct_local_sizes[ctxcnt-1] * CMS_val_tid_pair_SIZE;
  if(world_rank == 0) last_cct_size += cct_nzmids[ctxcnt - 1].size() * CMS_m_pair_SIZE;
  uint64_t total_size;
  getMyCCTs(cct_off,my_cct,last_cct_size,total_size, world_size, world_rank);
  updateCctOffset(cct_off,total_size, ctxcnt,threads/world_size);


  MPI_File thread_major_f;
  MPI_File_open(MPI_COMM_WORLD, (dir / "thread_major_sparse.db").c_str(),
                  MPI_MODE_RDONLY, MPI_INFO_NULL, &thread_major_f);
  MPI_File cct_major_f;
  MPI_File_open(MPI_COMM_WORLD, (dir / "cct_major_sparse.db").c_str(),
                  MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &cct_major_f);
  writeAsByte4(ctxcnt,cct_major_f,0);
  
  std::vector<ProfileInfo> prof_info;
  readProfileInfo(prof_info,thread_major_f);

  for(int i =0; i<my_cct.size(); i++){
    std::vector<uint32_t> cct_ids;
    cct_ids.emplace_back(my_cct[i]);
    rwOneCCTgroup(cct_ids,prof_info,cct_off,total_size, thread_major_f,cct_major_f);
  }

  MPI_File_close(&thread_major_f);
  MPI_File_close(&cct_major_f);



  

 
  
/* TEST cct_nzmids
  for(int i = 0; i<ctxcnt; i++){
    std::cout << "rank " << world_rank << "cct " << i << ":\n  ";
    for (std::set<uint16_t>::iterator it=cct_nzmids[i].begin(); it!=cct_nzmids[i].end(); ++it)
	    std::cout << ' ' << *it;
	  std::cout<<"\n";
  }
*/


/* TEST for collecting local sizes
  std::cout << "rank " << world_rank << ": (";
  for(auto& o:outputs.citerate()) std::cout << o.first->attributes.threadid() << " ";
  std::cout << ")\n";
  for(int i =0; i<ctxcnt; i++){
    std::cout << i << "--" << CCTID(i) << "--" <<cct_local_sizes[i] << "\n";
  }
*/
//TEST cct major functions
/*
  MPI_File thread_major_f;
  MPI_File_open(MPI_COMM_WORLD, (dir / "thread_major_sparse.db").c_str(),
                  MPI_MODE_RDONLY, MPI_INFO_NULL, &thread_major_f);
  std::vector<ProfileInfo> prof_info;
  readProfileInfo(prof_info,thread_major_f);
  
  std::vector<uint32_t> cct_ids;
  for(int i = 25; i<33; i++){
    cct_ids.emplace_back(i);
  }
  std::unordered_map<uint32_t,std::vector<DataBlock>> cct_data_pairs;
  rwOneCCTgroup(cct_ids,prof_info,cct_data_pairs,thread_major_f);
*/

/* TEMP: test some cct major functions
std::cout << "Rank " << world_rank << ": ";
  for(auto c :my_cct) std::cout << " " << c;
  std::cout << "\n\n";
  uint64_t last;
  printf("last cct %d ==? %d",my_cct[my_cct.size()-1],CCTID((ctxcnt - 1)));
  if(my_cct[my_cct.size()-1] == CCTID((ctxcnt - 1))){
    printf("I am here");
    last = last_cct_size;
    printf("last is %d\n", last);
  }else{
    last = cct_off[CCTLOCALSIZESIDX(my_cct[my_cct.size()-1]+2)].second - cct_off[CCTLOCALSIZESIDX(my_cct[my_cct.size()-1])].second;
  }
  printf("AAAAA %d:%d\n",cct_off[CCTLOCALSIZESIDX(my_cct[my_cct.size()-1])].second, cct_off[CCTLOCALSIZESIDX(my_cct[0])].second);


  std::cout << "size I am responsible for : " << (cct_off[CCTLOCALSIZESIDX(my_cct[my_cct.size()-1])].second - cct_off[CCTLOCALSIZESIDX(my_cct[0])].second +last) <<"\n";


  
  //TEST for binarySearchCCTid
  std::vector<std::pair<uint32_t, uint64_t>> profile_cct_offsets (30);
  for(int i =0;i<30;i++){
    profile_cct_offsets[i].first = i*2;
    profile_cct_offsets[i].second = i*10;
  }
  std::vector<uint32_t> cct_ids (5);
  cct_ids[0] = 29;
  cct_ids[1] = 30;
  cct_ids[2] = 31;
  cct_ids[3] = 32;
  cct_ids[4] = 36;
  std::vector<std::pair<uint32_t, uint64_t>> my_cct_offs;
  binarySearchCCTid(cct_ids,profile_cct_offsets,my_cct_offs);
  for(auto c :profile_cct_offsets) std::cout << c.first << ":" << c.second << " | ";
  std::cout << "\n";
  for(auto c :my_cct_offs) std::cout << c.first << ":" << c.second << " | ";
  std::cout << "\n";

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

template <typename T>
void SparseDB::exscan(std::vector<T>& data, int threads) {
  int n = data.size();
  int rounds = ceil(std::log2(n));
  std::vector<T> tmp (n);

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
