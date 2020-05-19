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

  exscan(tmp,threads);

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

void SparseDB::writeProfiles(std::vector<std::pair<uint32_t, uint64_t>>& prof_offsets, 
    std::vector<std::pair<const hpctoolkit::Thread*, uint64_t>>& profile_sizes, std::unordered_map<uint32_t,std::vector<char>>& prof_infos, 
    MPI_File fh, int threads){

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

    //extract bytes for profile information
    std::vector<char> info (TMS_prof_info_SIZE);
    std::copy(bytes.data(),bytes.data()+TMS_prof_skip_SIZE,info.begin());
    prof_infos.emplace(tid,info);


    //write at specific place
    MPI_Status stat;
    MPI_File_write_at(fh,my_prof_offset, bytes.data()+TMS_prof_skip_SIZE, bytes.size()-TMS_prof_skip_SIZE, MPI_BYTE, &stat);
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

  std::unordered_map<uint32_t,std::vector<char>> prof_infos;
  writeProfiles(prof_offsets, profile_sizes, prof_infos, thread_major_f, threads/world_size);
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
// [Profile informations] 
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

//input: local sizes for all cct, output: final offsets for all cct
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

//input: final offsets for all cct, output: my(rank) own responsible cct list
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
      interpretByte4(&cct_id, &input[i]);
      interpretByte8(&cct_off, &input[i+TMS_cct_id_SIZE]);
      cct_offsets.emplace_back(cct_id, cct_off);
    }
    
}

//given full cct offsets and a group of cct ids, return a vector of <cct id, offset> with nzcct
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
        //my_cct_offsets.emplace_back(target,0);
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
  my_cct_offsets.emplace_back(profile_cct_offsets[m+1].first,profile_cct_offsets[m+1].second);

  return 1;
}

void SparseDB::readOneProfile(std::vector<uint32_t>& cct_ids, ProfileInfo prof_info,
    std::unordered_map<uint32_t,std::vector<DataBlock>>& cct_data_pairs,MPI_File fh,MPI_Offset offset)
{
  //offset:beginning of the profile
  //cct_ids: the ccts it needs to read in for this round for this profile(ascending order)

  //TODO: val and mid write together => one seek per cct group

  //find the corresponding cct and its offset in values and mids
  MPI_Offset cct_offsets_offset = offset + prof_info.num_val * (TMS_val_SIZE + TMS_mid_SIZE);
  std::vector<std::pair<uint32_t,uint64_t>> full_cct_offsets (prof_info.num_nzcct);
  readCCToffsets(full_cct_offsets,fh,cct_offsets_offset);
  std::vector<std::pair<uint32_t,uint64_t>> my_cct_offsets;
  binarySearchCCTid(cct_ids,full_cct_offsets,my_cct_offsets);

  //read all values and metric ids for this group of cct at once
  MPI_Offset val_start_pos = offset + my_cct_offsets[0].second * TMS_val_SIZE;
  int val_count = (my_cct_offsets[my_cct_offsets.size()-1].second - my_cct_offsets[0].second) * TMS_val_SIZE;
  MPI_Offset mid_start_pos = offset + prof_info.num_val * TMS_val_SIZE + my_cct_offsets[0].second * TMS_mid_SIZE;
  int mid_count = (my_cct_offsets[my_cct_offsets.size()-1].second - my_cct_offsets[0].second) * TMS_mid_SIZE;
  char vinput[val_count];
  char minput[mid_count];
  MPI_Status stat;
  MPI_File_read_at(fh,val_start_pos,&vinput,val_count,MPI_BYTE,&stat);
  MPI_File_read_at(fh,mid_start_pos,&minput,mid_count,MPI_BYTE,&stat);

  //for each cct, keep track of the values,metric ids, and thread ids
  for(int c = 0; c<my_cct_offsets.size()-1; c++) 
  {
    uint32_t cct_id = my_cct_offsets[c].first;
    uint64_t num_val_this_cct = my_cct_offsets[c+1].second - my_cct_offsets[c].second;
    char* val_start_this_cct = vinput + (my_cct_offsets[c].second - my_cct_offsets[0].second) * TMS_val_SIZE;
    char* mid_start_this_cct = minput + (my_cct_offsets[c].second - my_cct_offsets[0].second) * TMS_mid_SIZE;
    for(int i = 0; i<num_val_this_cct; i++){
      //get a pair of val and mid
      uint64_t val;
      interpretByte8(&val,val_start_this_cct);
      uint16_t mid;
      interpretByte2(&mid,mid_start_this_cct);

      //store them
      std::unordered_map<uint32_t,std::vector<DataBlock>>::const_iterator got = cct_data_pairs.find (cct_id);
      if ( got == cct_data_pairs.end() ){
        //this cct doesn't exist in cct_data_paris yet
        
        DataBlock data;
        data.mid = mid;
        std::vector<std::pair<hpcrun_metricVal_t,uint32_t>> values_tids;
        hpcrun_metricVal_t v;
        v.r = val;
        values_tids.emplace_back(v,prof_info.tid);
        data.values_tids = values_tids;
        std::vector<DataBlock> datas;
        datas.emplace_back(data);
        cct_data_pairs.emplace(cct_id,datas);
      }else{
        
        std::vector<DataBlock> datas = got->second;
        auto it = std::find_if(datas.begin(), datas.end(), 
                       [mid] (const DataBlock& d) { 
                          return d.mid == mid; 
                       });


        hpcrun_metricVal_t v;
        v.r = val;
        if(it != datas.end()){
          it->values_tids.emplace_back(v,prof_info.tid);
        }else{
          DataBlock data;
          data.mid = mid;
          std::vector<std::pair<hpcrun_metricVal_t,uint32_t>> values_tids;
          values_tids.emplace_back(v,prof_info.tid);
          data.values_tids = values_tids;
          datas.emplace_back(data);
        }
      }

      
    }

  }

}



//***************************************************************************
// general - YUMENG
//***************************************************************************

void SparseDB::merge(int threads) {
  int world_rank;
  int world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

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
