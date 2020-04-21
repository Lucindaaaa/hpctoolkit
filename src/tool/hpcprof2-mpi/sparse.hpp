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

#define HPCTOOLKIT_PROF2MPI_SPARSE_H
#ifdef HPCTOOLKIT_PROF2MPI_SPARSE_H

#include "lib/profile/sink.hpp"
#include "lib/profile/stdshim/filesystem.hpp"
#include "lib/profile/util/once.hpp"
#include "lib/profile/util/locked_unordered.hpp"

#include <vector>
#include <mpi.h>

class SparseDB : public hpctoolkit::ProfileSink {
public:
  SparseDB(const hpctoolkit::stdshim::filesystem::path&);
  SparseDB(hpctoolkit::stdshim::filesystem::path&&);
  ~SparseDB() = default;

  void write() override;

  hpctoolkit::DataClass accepts() const noexcept override {
    using namespace hpctoolkit;
    return DataClass::threads | DataClass::contexts | DataClass::metrics;
  }

  hpctoolkit::DataClass wavefronts() const noexcept override {
    using namespace hpctoolkit;
    return DataClass::contexts;
  }

  hpctoolkit::ExtensionClass requires() const noexcept override {
    using namespace hpctoolkit;
    return ExtensionClass::identifier;
  }

  void notifyWavefront(hpctoolkit::DataClass::singleton_t) noexcept override;
  void notifyThreadFinal(const hpctoolkit::Thread::Temporary&) override;

  //YUMENG
  uint64_t getProfileSizes(std::vector<std::pair<const hpctoolkit::Thread*, uint64_t>>& profile_sizes);
  uint32_t getTotalNumProfiles(uint32_t my_num_prof);
  uint64_t getMyOffset(uint64_t my_size,int rank);
  void getMyProfOffset(std::vector<std::pair<uint32_t, uint64_t>>& prof_offsets,
      std::vector<std::pair<const hpctoolkit::Thread*, uint64_t>>& profile_sizes,
      uint32_t total_prof, uint64_t my_offset, int threads);
  void merge(int);
  void exscan(std::vector<uint64_t>& data,int threads); 

private:
  hpctoolkit::stdshim::filesystem::path dir;
  void merge0(int, MPI_File&, const std::vector<std::pair<hpctoolkit::ThreadAttributes,
    hpctoolkit::stdshim::filesystem::path>>&);
  void mergeN(int, MPI_File&);

  std::vector<std::reference_wrapper<const hpctoolkit::Context>> contexts;
  unsigned int ctxMaxId;
  hpctoolkit::util::OnceFlag contextPrep;
  void prepContexts() noexcept;

  hpctoolkit::util::locked_unordered_map<const hpctoolkit::Thread*,
    hpctoolkit::stdshim::filesystem::path> outputs;
  std::atomic<std::size_t> outputCnt;
};

#endif  // HPCTOOLKIT_PROF2MPI_SPARSE_H
