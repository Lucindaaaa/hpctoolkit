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
// Copyright ((c)) 2002-2020, Rice University
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

//***************************************************************************
//
// File:
//   $HeadURL$
//
// Purpose:
//   Low-level types and functions for reading/writing a call path
//   profile as formatted data.
//
//   See hpcrun-fmt.txt.
//
//   These routines *must not* allocate dynamic memory; if such memory
//   is needed, callbacks to the user's allocator should be used.
//
// Description:
//   [The set of functions, macros, etc. defined in the file]
//
//***************************************************************************

//************************* System Include Files ****************************

#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <sys/stat.h>

//*************************** User Include Files ****************************

#include <include/gcc-attr.h>

#include "hpcio.h"
#include "hpcio-buffer.h"
#include "hpcfmt.h"
#include "hpcrun-fmt.h"

//***************************************************************************

//***************************************************************************
// hdr
//***************************************************************************

int
hpcrun_fmt_hdr_fread(hpcrun_fmt_hdr_t* hdr, FILE* infs, hpcfmt_alloc_fn alloc)
{
  char tag[HPCRUN_FMT_MagicLen + 1];

  int nr = fread(tag, 1, HPCRUN_FMT_MagicLen, infs);
  tag[HPCRUN_FMT_MagicLen] = '\0';

  if (nr != HPCRUN_FMT_MagicLen) {
    return HPCFMT_ERR;
  }
  if (strcmp(tag, HPCRUN_FMT_Magic) != 0) {
    return HPCFMT_ERR;
  }

  nr = fread(hdr->versionStr, 1, HPCRUN_FMT_VersionLen, infs);
  hdr->versionStr[HPCRUN_FMT_VersionLen] = '\0';
  if (nr != HPCRUN_FMT_VersionLen) {
    return HPCFMT_ERR;
  }
  hdr->version = atof(hdr->versionStr);

  nr = fread(&hdr->endian, 1, HPCRUN_FMT_EndianLen, infs);
  if (nr != HPCRUN_FMT_EndianLen) {
    return HPCFMT_ERR;
  }

  hpcfmt_nvpairList_fread(&(hdr->nvps), infs, alloc);

  return HPCFMT_OK;
}


int
hpcrun_fmt_hdr_fwrite(FILE* fs, ...)
{
  va_list args;
  va_start(args, fs);

  fwrite(HPCRUN_FMT_Magic,   1, HPCRUN_FMT_MagicLen, fs);
  fwrite(HPCRUN_FMT_Version, 1, HPCRUN_FMT_VersionLen, fs);
  fwrite(HPCRUN_FMT_Endian,  1, HPCRUN_FMT_EndianLen, fs);

  hpcfmt_nvpairs_vfwrite(fs, args);

  va_end(args);

  return HPCFMT_OK;
}


int
hpcrun_fmt_hdr_fprint(hpcrun_fmt_hdr_t* hdr, FILE* fs)
{
  fprintf(fs, "%s\n", HPCRUN_FMT_Magic);

  fprintf(fs, "[hdr:\n");
  fprintf(fs, "  (version: %s)\n", hdr->versionStr);
  fprintf(fs, "  (endian: %c)\n", hdr->endian);
  hpcfmt_nvpairList_fprint(&hdr->nvps, fs, "  ");
  fprintf(fs, "]\n");

  return HPCFMT_OK;
}


void
hpcrun_fmt_hdr_free(hpcrun_fmt_hdr_t* hdr, hpcfmt_free_fn dealloc)
{
  if (dealloc) {
    hpcfmt_nvpairList_free(&(hdr->nvps), dealloc);
  }
}


//***************************************************************************
// epoch-hdr
//***************************************************************************

int
hpcrun_fmt_epochHdr_fread(hpcrun_fmt_epochHdr_t* ehdr, FILE* fs,
			  hpcfmt_alloc_fn alloc)
{

  char tag[HPCRUN_FMT_EpochTagLen + 1];

  int nr = fread(tag, 1, HPCRUN_FMT_EpochTagLen, fs);
  tag[HPCRUN_FMT_EpochTagLen] = '\0';

  if (nr != HPCRUN_FMT_EpochTagLen) {
    return (nr == 0 && feof(fs)) ? HPCFMT_EOF : HPCFMT_ERR;
  }

  if (strcmp(tag, HPCRUN_FMT_EpochTag) != 0) {
    return HPCFMT_ERR;
  }


  // removed m_raToCallsiteOfst from epoch Hdr. don't change file format!
  uint32_t dummy;

  HPCFMT_ThrowIfError(hpcfmt_int8_fread(&(ehdr->flags.bits), fs));
  HPCFMT_ThrowIfError(hpcfmt_int8_fread(&(ehdr->measurementGranularity), fs));
  HPCFMT_ThrowIfError(hpcfmt_int4_fread(&dummy, fs));
  HPCFMT_ThrowIfError(hpcfmt_nvpairList_fread(&(ehdr->nvps), fs, alloc));

  return HPCFMT_OK;

}


int
hpcrun_fmt_epochHdr_fwrite(FILE* fs, epoch_flags_t flags,
			   uint64_t measurementGranularity,
			   ...)
{
  va_list args;
  va_start(args, measurementGranularity);

  int nw = fwrite(HPCRUN_FMT_EpochTag, 1, HPCRUN_FMT_EpochTagLen, fs);
  if (nw != HPCRUN_FMT_EpochTagLen) return HPCFMT_ERR;

  // removed m_raToCallsiteOfst from epoch Hdr. don't change file format!
  uint32_t dummy = 0;

  HPCFMT_ThrowIfError(hpcfmt_int8_fwrite(flags.bits, fs));
  HPCFMT_ThrowIfError(hpcfmt_int8_fwrite(measurementGranularity, fs));
  HPCFMT_ThrowIfError(hpcfmt_int4_fwrite(dummy, fs));

  HPCFMT_ThrowIfError(hpcfmt_nvpairs_vfwrite(fs, args));

  va_end(args);

  return HPCFMT_OK;
}


int
hpcrun_fmt_epochHdr_fprint(hpcrun_fmt_epochHdr_t* ehdr, FILE* fs)
{
  fprintf(fs, "%s\n", HPCRUN_FMT_EpochTag);
  fprintf(fs, "[epoch-hdr:\n");
  fprintf(fs, "  (flags: 0x%"PRIx64")\n", ehdr->flags.bits);
  fprintf(fs, "  (measurement-granularity: %"PRIu64")\n",
	  ehdr->measurementGranularity);
  hpcfmt_nvpairList_fprint(&(ehdr->nvps), fs, "  ");
  fprintf(fs, "]\n");

  return HPCFMT_OK;
}


void
hpcrun_fmt_epochHdr_free(hpcrun_fmt_epochHdr_t* ehdr, hpcfmt_free_fn dealloc)
{
  if (dealloc) {
    hpcfmt_nvpairList_free(&(ehdr->nvps), dealloc);
  }
}


//***************************************************************************
// metric-tbl
//***************************************************************************

const metric_desc_t metricDesc_NULL = {
  .name          = NULL,
  .description   = NULL,
  .flags.bits_big[0] = 0,
  .flags.bits_big[1] = 0,
  .period        = 0,
  .properties = {.time = 0,.cycles = 0},
  .formula       = NULL,
  .format        = NULL,
};

const hpcrun_metricFlags_t hpcrun_metricFlags_NULL = {
  .fields.ty          = MetricFlags_Ty_NULL,
  .fields.valTy       = MetricFlags_ValTy_NULL,
  .fields.valFmt      = MetricFlags_ValFmt_NULL,
  .fields.unused0     = 0,

  .fields.partner     = 0,
  .fields.show        = (uint8_t)true,
  .fields.showPercent = (uint8_t)true,

  .fields.unused1     = 0,
};

hpcrun_metricVal_t hpcrun_metricVal_ZERO = { .bits = 0 };

//***************************************************************************

int
hpcrun_fmt_metricTbl_fread(metric_tbl_t* metric_tbl, metric_aux_info_t **aux_info,
		FILE* fs, double fmtVersion, hpcfmt_alloc_fn alloc)
{

  HPCFMT_ThrowIfError(hpcfmt_int4_fread(&(metric_tbl->len), fs));
  if (alloc) {
    metric_tbl->lst =
      (metric_desc_t*) alloc(metric_tbl->len * sizeof(metric_desc_t));
  }

  size_t aux_info_size = sizeof(metric_aux_info_t) * metric_tbl->len;
  metric_aux_info_t *perf_info = (metric_aux_info_t*)malloc(aux_info_size);
  memset(perf_info, 0, aux_info_size);

  for (uint32_t i = 0; i < metric_tbl->len; i++) {
    metric_desc_t* x = &metric_tbl->lst[i];
    HPCFMT_ThrowIfError(hpcrun_fmt_metricDesc_fread(x, &(perf_info)[i], fs, fmtVersion, alloc));
  }
  *aux_info = perf_info;

  return HPCFMT_OK;
}


int
hpcrun_fmt_metricTbl_fwrite(metric_desc_p_tbl_t* metric_tbl, metric_aux_info_t *aux_info, FILE* fs)
{
  for (uint32_t i = 0; i < metric_tbl->len; i++) {

	  // corner case: for other sampling sources than perf event, the
	  // value of aux_info is NULL

    metric_aux_info_t info_tmp;
	metric_aux_info_t *info_ptr = aux_info;

	if (aux_info == NULL) {
	  memset(&info_tmp, 0, sizeof(metric_aux_info_t));
	  info_ptr = &info_tmp;
	} else {
	  info_ptr = &(aux_info[i]);
	}

    hpcrun_fmt_metricDesc_fwrite(metric_tbl->lst[i], info_ptr, fs);
  }

  return HPCFMT_OK;
}


int
hpcrun_fmt_metricTbl_fprint(metric_tbl_t* metric_tbl, metric_aux_info_t *aux_info, FILE* fs)
{
  fprintf(fs, "[metric-tbl: (num-entries: %u)\n", metric_tbl->len);
  for (uint32_t i = 0; i < metric_tbl->len; i++) {
    metric_desc_t* x = &metric_tbl->lst[i];
    hpcrun_fmt_metricDesc_fprint(x, &(aux_info[i]), fs, "  ");
  }
  fputs("]\n", fs);

  return HPCFMT_OK;
}


void
hpcrun_fmt_metricTbl_free(metric_tbl_t* metric_tbl, hpcfmt_free_fn dealloc)
{
  for (uint32_t i = 0; i < metric_tbl->len; i++) {
    metric_desc_t* x = &metric_tbl->lst[i];
    hpcrun_fmt_metricDesc_free(x, dealloc);
  }
  dealloc((void*)metric_tbl->lst);
  metric_tbl->lst = NULL;
}


//***************************************************************************

int
hpcrun_fmt_metricDesc_fread(metric_desc_t* x, metric_aux_info_t *aux_info, FILE* fs,
			    double GCC_ATTR_UNUSED fmtVersion,
			    hpcfmt_alloc_fn alloc)
{
  HPCFMT_ThrowIfError(hpcfmt_str_fread(&(x->name), fs, alloc));
  HPCFMT_ThrowIfError(hpcfmt_str_fread(&(x->description), fs, alloc));
  HPCFMT_ThrowIfError(hpcfmt_intX_fread(x->flags.bits, sizeof(x->flags), fs));

  // FIXME: tallent: temporarily support old non-portable convention
  if ( !(x->flags.fields.ty == MetricFlags_Ty_Raw
	   || x->flags.fields.ty == MetricFlags_Ty_Final)
       || x->flags.fields.unused0 != 0
       || x->flags.fields.unused1 != 0) {
    fseek(fs, -sizeof(x->flags), SEEK_CUR);

    hpcrun_metricFlags_XXX_t x_flags_old;
    HPCFMT_ThrowIfError(hpcfmt_int8_fread(&(x_flags_old.bits[0]), fs));
    HPCFMT_ThrowIfError(hpcfmt_int8_fread(&(x_flags_old.bits[1]), fs));

    x->flags.bits_big[0] = 0;
    x->flags.bits_big[1] = 0;

    x->flags.fields.ty          = x_flags_old.fields.ty;
    x->flags.fields.valTy       = x_flags_old.fields.valTy;
    x->flags.fields.valFmt      = x_flags_old.fields.valFmt;
    x->flags.fields.partner     = (uint16_t) x_flags_old.fields.partner;
    x->flags.fields.show        = x_flags_old.fields.show;
    x->flags.fields.showPercent = x_flags_old.fields.showPercent;
  }

  HPCFMT_ThrowIfError(hpcfmt_int8_fread(&(x->period), fs));
  HPCFMT_ThrowIfError(hpcfmt_str_fread(&(x->formula), fs, alloc));
  HPCFMT_ThrowIfError(hpcfmt_str_fread(&(x->format), fs, alloc));

  HPCFMT_ThrowIfError(hpcfmt_int2_fread ((uint16_t*)&(x->is_frequency_metric),    fs));
  HPCFMT_ThrowIfError(hpcfmt_int2_fread ((uint16_t*)&(aux_info->is_multiplexed),  fs));
  HPCFMT_ThrowIfError(hpcfmt_real8_fread(&(aux_info->threshold_mean),  fs));
  HPCFMT_ThrowIfError(hpcfmt_int8_fread ((&aux_info->num_samples),     fs));

  // These two aren't written into the hpcrun file; hence manually set them.
  x->properties.time = 0;
  x->properties.cycles = 0;

  return HPCFMT_OK;
}


int
hpcrun_fmt_metricDesc_fwrite(metric_desc_t* x, metric_aux_info_t *aux_info, FILE* fs)
{
  HPCFMT_ThrowIfError(hpcfmt_str_fwrite(x->name, fs));
  HPCFMT_ThrowIfError(hpcfmt_str_fwrite(x->description, fs));
  HPCFMT_ThrowIfError(hpcfmt_intX_fwrite(x->flags.bits, sizeof(x->flags), fs));
  HPCFMT_ThrowIfError(hpcfmt_int8_fwrite(x->period, fs));
  HPCFMT_ThrowIfError(hpcfmt_str_fwrite(x->formula, fs));
  HPCFMT_ThrowIfError(hpcfmt_str_fwrite(x->format, fs));

  HPCFMT_ThrowIfError(hpcfmt_int2_fwrite(x->is_frequency_metric, fs));

  HPCFMT_ThrowIfError(hpcfmt_int2_fwrite(aux_info->is_multiplexed, fs));
  HPCFMT_ThrowIfError(hpcfmt_real8_fwrite(aux_info->threshold_mean, fs));
  HPCFMT_ThrowIfError(hpcfmt_int8_fwrite(aux_info->num_samples, fs));

  return HPCFMT_OK;
}


int
hpcrun_fmt_metricDesc_fprint(metric_desc_t* x, metric_aux_info_t *aux_info, FILE* fs, const char* pre)
{
  fprintf(fs, "%s[(nm: %s) (desc: %s) "
	  "((ty: %d) (val-ty: %d) (val-fmt: %d) (partner: %u) (show: %d) (showPercent: %d)) "
	  "(period: %"PRIu64") (formula: %s) (format: %s)\n" ,
	  pre, hpcfmt_str_ensure(x->name), hpcfmt_str_ensure(x->description),
	  (int)x->flags.fields.ty, (int)x->flags.fields.valTy,
	  (int)x->flags.fields.valFmt,
	  (uint)x->flags.fields.partner, x->flags.fields.show, x->flags.fields.showPercent,
	  x->period,
	  hpcfmt_str_ensure(x->formula), hpcfmt_str_ensure(x->format));
  fprintf(fs, "    (frequency: %d) (multiplexed: %d) (period-mean: %f) (num-samples: %d)]\n",
          (int)x->is_frequency_metric, (int)aux_info->is_multiplexed,
		  aux_info->threshold_mean,  (int) aux_info->num_samples);
  return HPCFMT_OK;
}


void
hpcrun_fmt_metricDesc_free(metric_desc_t* x, hpcfmt_free_fn dealloc)
{
  hpcfmt_str_free(x->name, dealloc);
  x->name = NULL;
  hpcfmt_str_free(x->description, dealloc);
  x->description = NULL;
  hpcfmt_str_free(x->formula, dealloc);
  x->formula = NULL;
  hpcfmt_str_free(x->format, dealloc);
  x->format = NULL;
}


void
hpcrun_fmt_metric_set_format(metric_desc_t *metric_desc, char *format)
{
  metric_desc->format = format;
}



double
hpcrun_fmt_metric_get_value(metric_desc_t metric_desc, hpcrun_metricVal_t metric)
{
  if (metric_desc.flags.fields.valFmt == MetricFlags_ValFmt_Int) {
    return (double) metric.i;
  }
  else if (metric_desc.flags.fields.valFmt == MetricFlags_ValFmt_Real) {
    return metric.r;
  }
  // TODO: default value
  return metric.r;
}

// set a new value into a metric
void
hpcrun_fmt_metric_set_value(metric_desc_t metric_desc,
   hpcrun_metricVal_t *metric, double value)
{
  if (metric_desc.flags.fields.valFmt == MetricFlags_ValFmt_Int) {
    metric->i = (int) value;
  }
  else if (metric_desc.flags.fields.valFmt == MetricFlags_ValFmt_Real) {
    metric->r = value;
  }
}

// set a new integer value into a metric, and force it to be
// an integer type metric
void
hpcrun_fmt_metric_set_value_int( hpcrun_metricFlags_t *flags,
   hpcrun_metricVal_t *metric, int value)
{
  flags->fields.valFmt = MetricFlags_ValFmt_Int;
  metric->i = value;
}

// set a new double value into a metric, and force it to be
// a real type metric
void
hpcrun_fmt_metric_set_value_real( hpcrun_metricFlags_t *flags,
   hpcrun_metricVal_t *metric, double value)
{
  flags->fields.valFmt = MetricFlags_ValFmt_Real;
  metric->r = value;
}


//***************************************************************************
// loadmap
//***************************************************************************

int
hpcrun_fmt_loadmap_fread(loadmap_t* loadmap, FILE* fs, hpcfmt_alloc_fn alloc)
{

#if 1
//YUMENG: no epoch, so loadmap needs to handle EOF situation
  int r = hpcfmt_int4_fread(&(loadmap->len), fs);
  if(r == HPCFMT_EOF ){
    return HPCFMT_EOF;
  }
  if(r != HPCFMT_OK){
    return HPCFMT_ERR;
  }
#else
  HPCFMT_ThrowIfError(hpcfmt_int4_fread(&(loadmap->len), fs));
#endif

  if (alloc) {
    loadmap->lst = alloc(loadmap->len * sizeof(loadmap_entry_t));
  }
  for (uint32_t i = 0; i < loadmap->len; i++) {
    loadmap_entry_t* e = &loadmap->lst[i];
    int ret = hpcrun_fmt_loadmapEntry_fread(e, fs, alloc);
    if(ret == HPCFMT_ERR) return HPCFMT_ERR;
  }

  return HPCFMT_OK;
}


int
hpcrun_fmt_loadmap_fwrite(loadmap_t* loadmap, FILE* fs)
{
  HPCFMT_ThrowIfError(hpcfmt_int4_fwrite(loadmap->len, fs));
  for (uint32_t i = 0; i < loadmap->len; i++) {
    loadmap_entry_t* e = &loadmap->lst[i];
    HPCFMT_ThrowIfError(hpcrun_fmt_loadmapEntry_fwrite(e, fs));
  }

  return HPCFMT_OK;
}


int
hpcrun_fmt_loadmap_fprint(loadmap_t* loadmap, FILE* fs)
{
  fprintf(fs, "[loadmap: (num-entries: %u)\n", loadmap->len);
  for (uint32_t i = 0; i < loadmap->len; i++) {
    loadmap_entry_t* e = &loadmap->lst[i];
    HPCFMT_ThrowIfError(hpcrun_fmt_loadmapEntry_fprint(e, fs, "  "));
  }
  fprintf(fs, "]\n");

  return HPCFMT_OK;
}


void
hpcrun_fmt_loadmap_free(loadmap_t* loadmap, hpcfmt_free_fn dealloc)
{
  for (uint32_t i = 0; i < loadmap->len; i++) {
    loadmap_entry_t* e = &loadmap->lst[i];
    hpcrun_fmt_loadmapEntry_free(e, dealloc);
  }
  dealloc(loadmap->lst);
  loadmap->lst = NULL;
}


//***************************************************************************

int
hpcrun_fmt_loadmapEntry_fread(loadmap_entry_t* x, FILE* fs,
			      hpcfmt_alloc_fn alloc)
{
  int ret = 10;
  HPCFMT_ThrowIfError(hpcfmt_int2_fread(&(x->id), fs));
  HPCFMT_ThrowIfError(hpcfmt_str_fread(&(x->name), fs, alloc));
  ret += (strlen(x->name) + 4);
  HPCFMT_ThrowIfError(hpcfmt_int8_fread(&(x->flags), fs));
  //return HPCFMT_OK;
  return ret;
}


int
hpcrun_fmt_loadmapEntry_fwrite(loadmap_entry_t* x, FILE* fs)
{
  HPCFMT_ThrowIfError(hpcfmt_int2_fwrite(x->id, fs));
  HPCFMT_ThrowIfError(hpcfmt_str_fwrite(x->name, fs));
  HPCFMT_ThrowIfError(hpcfmt_int8_fwrite(x->flags, fs));
  return HPCFMT_OK;
}


int
hpcrun_fmt_loadmapEntry_fprint(loadmap_entry_t* x, FILE* fs, const char* pre)
{
  fprintf(fs, "%s[(id: %u) (nm: %s) (flg: 0x%"PRIx64")]\n",
	  pre, (uint)x->id, x->name, x->flags);
  return HPCFMT_OK;
}


void
hpcrun_fmt_loadmapEntry_free(loadmap_entry_t* x, hpcfmt_free_fn dealloc)
{
  hpcfmt_str_free(x->name, dealloc);
  x->name = NULL;
}


//***************************************************************************
// cct
//***************************************************************************

 int
hpcrun_fmt_cct_node_fread(hpcrun_fmt_cct_node_t* x,
			  epoch_flags_t flags, FILE* fs)
{
  HPCFMT_ThrowIfError(hpcfmt_int4_fread(&x->id, fs));
  HPCFMT_ThrowIfError(hpcfmt_int4_fread(&x->id_parent, fs));

  x->as_info = lush_assoc_info_NULL;
  if (flags.fields.isLogicalUnwind) {
    HPCFMT_ThrowIfError(hpcfmt_int4_fread(&x->as_info.bits, fs));
  }

  HPCFMT_ThrowIfError(hpcfmt_int2_fread(&x->lm_id, fs));
  HPCFMT_ThrowIfError(hpcfmt_int8_fread(&x->lm_ip, fs));

  lush_lip_init(&x->lip);
  if (flags.fields.isLogicalUnwind) {
    hpcrun_fmt_lip_fread(&x->lip, fs);
  }
  for (int i = 0; i < x->num_metrics; ++i) {
    HPCFMT_ThrowIfError(hpcfmt_int8_fread(&x->metrics[i].bits, fs));
  }
  return HPCFMT_OK;
}


int
hpcrun_fmt_cct_node_fwrite(hpcrun_fmt_cct_node_t* x,
			   epoch_flags_t flags, FILE* fs)
{
  HPCFMT_ThrowIfError(hpcfmt_int4_fwrite(x->id, fs));
  HPCFMT_ThrowIfError(hpcfmt_int4_fwrite(x->id_parent, fs));

  if (flags.fields.isLogicalUnwind) {
    HPCFMT_ThrowIfError(hpcfmt_int4_fwrite(x->as_info.bits, fs));
  }

  HPCFMT_ThrowIfError(hpcfmt_int2_fwrite(x->lm_id, fs));
  HPCFMT_ThrowIfError(hpcfmt_int8_fwrite(x->lm_ip, fs));

  if (flags.fields.isLogicalUnwind) {
    HPCFMT_ThrowIfError(hpcrun_fmt_lip_fwrite(&x->lip, fs));
  }
  for (int i = 0; i < x->num_metrics; ++i) {
    HPCFMT_ThrowIfError(hpcfmt_int8_fwrite(x->metrics[i].bits, fs));
  }

  return HPCFMT_OK;
}


#if 0
int
hpcrun_fmt_cct_node_fprint(hpcrun_fmt_cct_node_t* x, FILE* fs,
			   epoch_flags_t flags, const metric_tbl_t* metricTbl,
			   const char* pre)
#else
//YUMENG: no need to parse metricTbl for sparse format
int
hpcrun_fmt_cct_node_fprint(hpcrun_fmt_cct_node_t* x, FILE* fs,
			   epoch_flags_t flags, const char* pre)
#endif
{
  // N.B.: convert 'id' and 'id_parent' to ints so leaf flag
  // (negative) is apparent
  fprintf(fs, "%s[node: (id: %d) (id-parent: %d) ",
	  pre, (int)x->id, (int)x->id_parent);

  if (flags.fields.isLogicalUnwind) {
    char as_str[LUSH_ASSOC_INFO_STR_MIN_LEN];
    lush_assoc_info_sprintf(as_str, x->as_info);

    fprintf(fs, "(as: %s) ", as_str);
  }

  fprintf(fs, "(lm-id: %u) (lm-ip: 0x%"PRIx64") ", (uint)x->lm_id, x->lm_ip);

  if (flags.fields.isLogicalUnwind) {
    hpcrun_fmt_lip_fprint(&x->lip, fs, "");
  }

//YUMENG: no need for sparse format
#if 0
  fprintf(fs, "\n");

  fprintf(fs, "%s(metrics:", pre);
  for (uint i = 0; i < x->num_metrics; ++i) {
    hpcrun_metricFlags_t mflags = hpcrun_metricFlags_NULL;
    if (metricTbl) {
      const metric_desc_t* mdesc = &(metricTbl->lst[i]);
      mflags = mdesc->flags;
    }


    switch (mflags.fields.valFmt) {
      default:
      case MetricFlags_ValFmt_Int:
	fprintf(fs, " %"PRIu64, x->metrics[i].i);
	break;
      case MetricFlags_ValFmt_Real:
	fprintf(fs, " %g", x->metrics[i].r);
	break;
    }

    if (i + 1 < x->num_metrics) {
      fprintf(fs, " ");
    }
  }
  fprintf(fs, ")\n");
#endif
  fprintf(fs, "%s]\n", pre);

  return HPCFMT_OK;
}


//***************************************************************************
// sparse metircs - YUMENG
//***************************************************************************

 int
hpcrun_fmt_sparse_metrics_fread(hpcrun_fmt_sparse_metrics_t* x, FILE* fs)
{

  HPCFMT_ThrowIfError(hpcfmt_int4_fread(&(x->tid), fs));
  HPCFMT_ThrowIfError(hpcfmt_int8_fread(&(x->num_vals), fs));

  x->values = (hpcrun_metricVal_t *) malloc((x->num_vals)*sizeof(hpcrun_metricVal_t));
  for (int i = 0; i < x->num_vals; ++i) {
    HPCFMT_ThrowIfError(hpcfmt_int8_fread(&(x->values[i].bits), fs));
  }

  x->mid = (uint16_t *) malloc((x->num_vals)*sizeof(uint16_t));
  //x->m_offset = (uint64_t *) malloc((x->num_vals)*sizeof(uint64_t));
  for (int i = 0; i < x->num_vals; ++i) {
    HPCFMT_ThrowIfError(hpcfmt_int2_fread(&x->mid[i], fs));
    //HPCFMT_ThrowIfError(hpcfmt_int8_fread(&x->m_offset[i], fs));
  }

  HPCFMT_ThrowIfError(hpcfmt_int4_fread(&(x->num_nz_cct), fs));
  x->cct_id = (uint32_t *) malloc((x->num_nz_cct)*sizeof(uint32_t));
  x->cct_off = (uint64_t *) malloc((x->num_nz_cct)*sizeof(uint64_t));
  for (int i = 0; i < x->num_nz_cct; ++i) {
    HPCFMT_ThrowIfError(hpcfmt_int4_fread(&x->cct_id[i], fs));
    HPCFMT_ThrowIfError(hpcfmt_int8_fread(&x->cct_off[i], fs));
  }

  return HPCFMT_OK;
}


int
hpcrun_fmt_sparse_metrics_fwrite(hpcrun_fmt_sparse_metrics_t* x,FILE* fs)
{
  HPCFMT_ThrowIfError(hpcfmt_int4_fwrite((uint32_t)x->tid, fs));
  HPCFMT_ThrowIfError(hpcfmt_int8_fwrite(x->num_vals, fs));


  for (int i = 0; i < x->num_vals; ++i) {
    HPCFMT_ThrowIfError(hpcfmt_int8_fwrite(x->values[i].bits, fs));
  }

  for (int i = 0; i < x->num_vals; ++i) {
    HPCFMT_ThrowIfError(hpcfmt_int2_fwrite(x->mid[i], fs));
    //YUMENG: TODO: no need to collect info at first place, for now just stop writing it
    //HPCFMT_ThrowIfError(hpcfmt_int8_fwrite(x->m_offset[i], fs));
    
  }

  HPCFMT_ThrowIfError(hpcfmt_int4_fwrite((uint32_t)x->num_nz_cct, fs));

  for (int i = 0; i < x->num_nz_cct; ++i) {
    HPCFMT_ThrowIfError(hpcfmt_int4_fwrite((uint32_t)x->cct_id[i], fs));
    HPCFMT_ThrowIfError(hpcfmt_int8_fwrite((uint64_t)x->cct_off[i], fs));
  }

  return HPCFMT_OK;
}


int
hpcrun_fmt_sparse_metrics_fprint(hpcrun_fmt_sparse_metrics_t* x, FILE* fs,
          const metric_tbl_t* metricTbl, const char* pre)
{
  fprintf(fs, "[sparse metrics:\n");
  fprintf(fs, "%s(thread ID: %d)\n%s(number of non-zero metrics: %d)\n",
	  pre, (int)x->tid,pre, (int)x->num_vals);

  fprintf(fs, "%s(values: ", pre);
  for (uint i = 0; i < x->num_vals; ++i) {
    hpcrun_metricFlags_t mflags = hpcrun_metricFlags_NULL;
    if (metricTbl) {
      int metric_id = x->mid[i];
      const metric_desc_t* mdesc = &(metricTbl->lst[metric_id]);
      mflags = mdesc->flags;
    }

    switch (mflags.fields.valFmt) {
      default: //used for printing tmp sparse-db files, it does not have metric_tbl included
        fprintf(fs, " %g", x->values[i].r);
	      break;
      case MetricFlags_ValFmt_Int:
	      fprintf(fs, " %"PRIu64, x->values[i].i);
      	break;
      case MetricFlags_ValFmt_Real:
	      fprintf(fs, " %g", x->values[i].r);
	      break;
    }

    if (i + 1 < x->num_vals) {
      fprintf(fs, " ");
    }
  }
  fprintf(fs, ")\n");

  fprintf(fs, "%s(metric id: ",pre);
  for (uint i = 0; i < x->num_vals; ++i) {
    //fprintf(fs, "%d:%d", x->mid[i],x->m_offset[i]);
    fprintf(fs, "%d", x->mid[i]);
    if (i + 1 < x->num_vals) {
      fprintf(fs, " ");
    }
  }
  fprintf(fs, ")\n");

  fprintf(fs,"%s(cct offsets (cct id : offset): ",pre);
  for (uint i = 0; i < x->num_nz_cct; i++) {
    fprintf(fs, "%d:%d", x->cct_id[i], x->cct_off[i]);
    if (i + 1 < x->num_nz_cct) {
      fprintf(fs, " ");
    }
  }
  fprintf(fs, ")\n");

  fprintf(fs, "]\n");

  return HPCFMT_OK;
}

//***************************************************************************
// hpcrun_sparse_file - YUMENG
//***************************************************************************
hpcrun_sparse_file_t* hpcrun_sparse_open(const char* path)
{
  FILE* fs = hpcio_fopen_r(path);
  if(!fs) return NULL;

  hpcrun_sparse_file_t* sparse_fs = (hpcrun_sparse_file_t*) malloc(sizeof(hpcrun_sparse_file_t));
  sparse_fs->file = fs;
  sparse_fs->mode = OPENED;
  sparse_fs->cur_pos = 0;

  sparse_fs->cur_cct = 0;
  sparse_fs->cur_metric = 0;
  sparse_fs->cur_metric_id = 0;
  sparse_fs->cur_lm = 0;
  sparse_fs->cur_block = 0;

  //initialize footer
  fseek(fs, 0, SEEK_END);
  size_t footer_position = ftell(fs) - 56;
  fseek(fs, footer_position, SEEK_SET);
  for(int i = 0; i<7; i++){
    if( hpcfmt_int8_fread(&(sparse_fs->footer[i]), fs) != HPCFMT_OK) {
      free(sparse_fs);
      return NULL;
    }
  }
  fseek(fs, sparse_fs->footer[0], SEEK_SET);

  return sparse_fs;
}

/* succeed: return 0; fail: return 1; */
int hpcrun_sparse_pause(hpcrun_sparse_file_t* sparse_fs)
{
  sparse_fs->cur_pos = ftell(sparse_fs->file);
  int ret = hpcio_fclose(sparse_fs->file);
  if(!ret) sparse_fs->mode = PAUSED;
  return ret;
}

/* succeed: return 0; fail open: return 1; was open already: return -1 */
int hpcrun_sparse_resume(hpcrun_sparse_file_t* sparse_fs, const char* path)
{
  if(sparse_fs->mode == OPENED){
    fprintf(stderr, "ERROR: the file descriptor in the hpcrun_sparse_file object is still open\n");
    //exit(-1); //maybe change to not exit directly?
    return HPCFMT_ERR;
  }
  FILE* fs = hpcio_fopen_r(path);
  if(!fs) return 1;
  sparse_fs->file = fs;
  fseek(fs, sparse_fs->cur_pos, SEEK_SET);
  sparse_fs->mode = OPENED;
  return 0;
}

void hpcrun_sparse_close(hpcrun_sparse_file_t* sparse_fs)
{
  if(sparse_fs->mode == OPENED) hpcio_fclose(sparse_fs->file);
  free(sparse_fs);
}

/* succeed: returns 0; error while reading: returns -1 */
int hpcrun_sparse_read_hdr(hpcrun_sparse_file_t* sparse_fs, hpcrun_fmt_hdr_t* hdr)
{
  fseek(sparse_fs->file, sparse_fs->footer[0],SEEK_SET);
  int ret = hpcrun_fmt_hdr_fread(hdr, sparse_fs->file, malloc);
  if(ret != HPCFMT_OK) return HPCFMT_ERR;
  return 0;
}

/* succeed: returns positive id; End of list: returns 0; Fail reading: returns -1; */
int hpcrun_sparse_next_lm(hpcrun_sparse_file_t* sparse_fs, loadmap_entry_t* lm)
{
  if(sparse_fs->cur_lm == 0) sparse_fs->cur_lm = 4; //the first lm should skip the info about number of lms

  size_t realoffset = sparse_fs->footer[1] + sparse_fs->cur_lm;
  if(realoffset == sparse_fs->footer[3]) return 0; // no more next lm
  fseek(sparse_fs->file, realoffset, SEEK_SET);
  int ret = hpcrun_fmt_loadmapEntry_fread(lm, sparse_fs->file, malloc);
  sparse_fs->cur_lm += ret;

  return (lm) ? lm->id : HPCFMT_ERR;
}

/* succeed: returns a metric ID; end of list: returns 0; error: returns -1 */
int hpcrun_sparse_next_metric(hpcrun_sparse_file_t* sparse_fs, metric_desc_t* m, metric_aux_info_t* perf_info,double fmtVersion)
{
  if(sparse_fs->cur_metric == 0) sparse_fs->cur_metric = 4; //the first metric should skip the info about number of metrics

  size_t realoffset = sparse_fs->footer[4] + sparse_fs->cur_metric;
  if(realoffset == sparse_fs->footer[5]) return 0; // no more next metric
  fseek(sparse_fs->file, realoffset, SEEK_SET);
  HPCFMT_ThrowIfError(hpcrun_fmt_metricDesc_fread(m, perf_info, sparse_fs->file, fmtVersion, malloc));
  sparse_fs->cur_metric_id += 1;
  sparse_fs->cur_metric += (ftell(sparse_fs->file) - realoffset);

  return sparse_fs->cur_metric_id;
}

/* succeed: returns a cct ID; end of list: returns 0; error: returns -1 */
int hpcrun_sparse_next_context(hpcrun_sparse_file_t* sparse_fs, hpcrun_fmt_cct_node_t* node)
{
  if(sparse_fs->cur_cct == sparse_fs->footer[2]) return 0; //no more cct
  size_t realoffset = sparse_fs->footer[3] + (18 * sparse_fs->cur_cct) + 8; //8 is the info about number of ccts
  fseek(sparse_fs->file, realoffset, SEEK_SET);
  epoch_flags_t fake = {0};//need to remove in the future
  node->num_metrics = 0;
  HPCFMT_ThrowIfError(hpcrun_fmt_cct_node_fread(node, fake, sparse_fs->file));
  sparse_fs->cur_cct ++;
  return node->id;
}

/* succeed: returns a cct ID that we can read next_entry for; end of list: returns 0; error: returns -1 */
int hpcrun_sparse_next_block(hpcrun_sparse_file_t* sparse_fs)
{
  if(sparse_fs->cur_block == 0){
    fseek(sparse_fs->file,(sparse_fs->footer[5]+4),SEEK_SET);
    HPCFMT_ThrowIfError(hpcfmt_int8_fread(&(sparse_fs->num_nzval),sparse_fs->file));
    fseek(sparse_fs->file,(sparse_fs->footer[5]+12+10*(sparse_fs->num_nzval)),SEEK_SET);
    HPCFMT_ThrowIfError(hpcfmt_int4_fread(&(sparse_fs->num_nz_cct),sparse_fs->file));
    sparse_fs->val_offset = sparse_fs->footer[5]+ 12;
    sparse_fs->metric_pos_offset = sparse_fs->val_offset + 8*(sparse_fs->num_nzval);
    sparse_fs->cct_offset_offset = sparse_fs->metric_pos_offset + 2*(sparse_fs->num_nzval) +4; //change 10 to 2 for m_offset removal
  }
  if(sparse_fs->cur_block == sparse_fs->num_nz_cct) return 0; //no more cct block

  //seek to the first metric_position(id:offset) related to this cct
  fseek(sparse_fs->file,(sparse_fs->cct_offset_offset + 12*sparse_fs->cur_block),SEEK_SET);
  size_t metric_block_pos;
  int id;
  HPCFMT_ThrowIfError(hpcfmt_int4_fread(&id,sparse_fs->file));
  HPCFMT_ThrowIfError(hpcfmt_int8_fread(&metric_block_pos,sparse_fs->file));
  fseek(sparse_fs->file,4,SEEK_CUR);
  HPCFMT_ThrowIfError(hpcfmt_int8_fread(&(sparse_fs->cur_block_end),sparse_fs->file));
  if(sparse_fs->cur_block == sparse_fs->num_nz_cct - 1) sparse_fs->cur_block_end = sparse_fs->num_nzval;
  if(metric_block_pos == sparse_fs->num_nzval) return 0; //might not be end of cct, but rest blocks are all empty
  fseek(sparse_fs->file,(sparse_fs->metric_pos_offset + 2*metric_block_pos),SEEK_SET); //change 10 to 2 for m_offset removal
  sparse_fs->cur_block++;

  return id;

}

/* succeed: returns positive metricID (matching metricTbl, start from 1); end of this block: 0;error: return -1*/
/* ASSUMPTION: it is called continously for one block, i.e. no other fseek happen between calls */
int hpcrun_sparse_next_entry(hpcrun_sparse_file_t* sparse_fs, hpcrun_metricVal_t* val)
{
  if(sparse_fs->cur_block == 0){
    printf("ERROR: hpcrun_sparse_next_entry() has to be called after hpcrun_sparse_next_block() to set up entry point.\n");
    return HPCFMT_ERR;
  }
  size_t cur_pos = ftell(sparse_fs->file);
  if(cur_pos == sparse_fs->metric_pos_offset + 2*(sparse_fs->cur_block_end) )return 0; //change 10 to 2 for m_offset removal

  uint16_t id;
  uint64_t offset;
  HPCFMT_ThrowIfError(hpcfmt_int2_fread(&id,sparse_fs->file));
  //HPCFMT_ThrowIfError(hpcfmt_int8_fread(&offset,sparse_fs->file));
  id ++; //match the metric id in metricTbl(starting as 1), it was recorded starting as 0
  offset = (cur_pos - sparse_fs->metric_pos_offset)/2; //for one thread, each metric id only has one value

  //If MULTIPLE THREADS, then it will be a loop to get all values for one metric id
  fseek(sparse_fs->file,(sparse_fs->val_offset)+8*offset,SEEK_SET);
  HPCFMT_ThrowIfError(hpcfmt_int8_fread(&(val->bits),sparse_fs->file));
  fseek(sparse_fs->file,cur_pos+2,SEEK_SET); //set for next entry //change 10 to 2 for m_offset removal

  return id;
}


//***************************************************************************
// thread_major_sparse.db helper - YUMENG
//***************************************************************************
int
tms_thread_offset_fwrite(uint32_t num_t,uint64_t* x, FILE* fs)
{
  HPCFMT_ThrowIfError(hpcfmt_int4_fwrite(num_t, fs));

  for (int i = 0; i < num_t; ++i) {
    HPCFMT_ThrowIfError(hpcfmt_int8_fwrite(x[i], fs));
  }
  return HPCFMT_OK;
}

uint64_t* 
tms_thread_offset_fread(uint32_t* num_t,FILE* fs)
{
  HPCFMT_ThrowIfError(hpcfmt_int4_fread(num_t, fs));

  uint64_t* x = (uint64_t *) malloc((*num_t)*sizeof(uint64_t));

  for (int i = 0; i < *num_t; ++i) {
    HPCFMT_ThrowIfError(hpcfmt_int8_fread(&x[i], fs));
  }
  return x;
}

void
tms_thread_offset_fprint(uint32_t num_t,uint64_t* x, FILE* fs)
{
  fprintf(fs,"[Threads offsets (thread id : offset)\n  (");

  for (int i = 0; i < num_t; ++i) {
    fprintf(fs,"%d:%d ",i,x[i]);
  }
  fprintf(fs,")\n]\n");
  return HPCFMT_OK;
}

//***************************************************************************

int
hpcrun_fmt_lip_fread(lush_lip_t* x, FILE* fs)
{
  for (int i = 0; i < LUSH_LIP_DATA8_SZ; ++i) {
    HPCFMT_ThrowIfError(hpcfmt_int8_fread(&x->data8[i], fs));
  }

  return HPCFMT_OK;
}


int
hpcrun_fmt_lip_fwrite(lush_lip_t* x, FILE* fs)
{
  for (int i = 0; i < LUSH_LIP_DATA8_SZ; ++i) {
    HPCFMT_ThrowIfError(hpcfmt_int8_fwrite(x->data8[i], fs));
  }

  return HPCFMT_OK;
}


int
hpcrun_fmt_lip_fprint(lush_lip_t* x, FILE* fs, const char* pre)
{
  char lip_str[LUSH_LIP_STR_MIN_LEN];
  lush_lip_sprintf(lip_str, x);

  fprintf(fs, "%s(lip: %s)", pre, lip_str);

  return HPCFMT_OK;
}


//***************************************************************************
// hpctrace (located here for now)
//***************************************************************************

//***************************************************************************
// [hpctrace] hdr
//***************************************************************************

const hpctrace_hdr_flags_t hpctrace_hdr_flags_NULL = 0;


int
hpctrace_fmt_hdr_fread(hpctrace_fmt_hdr_t* hdr, FILE* infs)
{
  char tag[HPCTRACE_FMT_MagicLen + 1];

  int nr = fread(tag, 1, HPCTRACE_FMT_MagicLen, infs);
  tag[HPCTRACE_FMT_MagicLen] = '\0';

  if (nr != HPCTRACE_FMT_MagicLen) {
    return HPCFMT_ERR;
  }
  if (strcmp(tag, HPCTRACE_FMT_Magic) != 0) {
    return HPCFMT_ERR;
  }

  nr = fread(hdr->versionStr, 1, HPCTRACE_FMT_VersionLen, infs);
  hdr->versionStr[HPCTRACE_FMT_VersionLen] = '\0';
  if (nr != HPCTRACE_FMT_VersionLen) {
    return HPCFMT_ERR;
  }
  hdr->version = atof(hdr->versionStr);

  nr = fread(&hdr->endian, 1, HPCTRACE_FMT_EndianLen, infs);
  if (nr != HPCTRACE_FMT_EndianLen) {
    return HPCFMT_ERR;
  }

  hdr->flags = hpctrace_hdr_flags_NULL;
  if (hdr->version > 1.0) {
    HPCFMT_ThrowIfError(hpcfmt_int8_fread(&(hdr->flags), infs));
  }

  return HPCFMT_OK;
}


// Writer based on outbuf.
// Returns: HPCFMT_OK on success, else HPCFMT_ERR.
int
hpctrace_fmt_hdr_outbuf(hpctrace_hdr_flags_t flags, hpcio_outbuf_t* outbuf)
{
  ssize_t ret;

  const int bufSZ = sizeof(flags);
  unsigned char buf[bufSZ];

  uint64_t flag_bits = flags;
  int k = 0;
  for (int shift = 56; shift >= 0; shift -= 8) {
    buf[k] = (flag_bits >> shift) & 0xff;
    k++;
  }

  hpcio_outbuf_write(outbuf, HPCTRACE_FMT_Magic, HPCTRACE_FMT_MagicLen);
  hpcio_outbuf_write(outbuf, HPCTRACE_FMT_Version, HPCTRACE_FMT_VersionLen);
  hpcio_outbuf_write(outbuf, HPCTRACE_FMT_Endian, HPCTRACE_FMT_EndianLen);
  ret = hpcio_outbuf_write(outbuf, buf, bufSZ);

  if (ret != bufSZ) {
    return HPCFMT_ERR;
  }

  return HPCFMT_OK;
}


// N.B.: not async safe
int
hpctrace_fmt_hdr_fwrite(hpctrace_hdr_flags_t flags, FILE* fs)
{
  int nw;

  nw = fwrite(HPCTRACE_FMT_Magic,   1, HPCTRACE_FMT_MagicLen, fs);
  if (nw != HPCTRACE_FMT_MagicLen) return HPCFMT_ERR;

  nw = fwrite(HPCTRACE_FMT_Version, 1, HPCTRACE_FMT_VersionLen, fs);
  if (nw != HPCTRACE_FMT_VersionLen) return HPCFMT_ERR;

  nw = fwrite(HPCTRACE_FMT_Endian,  1, HPCTRACE_FMT_EndianLen, fs);
  if (nw != HPCTRACE_FMT_EndianLen) return HPCFMT_ERR;

  HPCFMT_ThrowIfError(hpcfmt_int8_fwrite(flags, fs));

  return HPCFMT_OK;
}


int
hpctrace_fmt_hdr_fprint(hpctrace_fmt_hdr_t* hdr, FILE* fs)
{
  fprintf(fs, "%s\n", HPCTRACE_FMT_Magic);

  fprintf(fs, "[hdr:\n");
  fprintf(fs, "  (version: %s)\n", hdr->versionStr);
  fprintf(fs, "  (endian: %c)\n", hdr->endian);
  fprintf(fs, "  (flags: 0x%"PRIx64")\n", hdr->flags);
  fprintf(fs, "]\n");

  return HPCFMT_OK;
}


//***************************************************************************
// [hpctrace] datum (trace record)
//***************************************************************************

int
hpctrace_fmt_datum_fread(hpctrace_fmt_datum_t* x, hpctrace_hdr_flags_t flags,
			 FILE* fs)
{
  int ret = HPCFMT_OK;

  ret = hpcfmt_int8_fread(&(x->comp), fs);
  if (ret != HPCFMT_OK) {
    return ret; // can be HPCFMT_EOF
  }

  HPCFMT_ThrowIfError(hpcfmt_int4_fread(&(x->cpId), fs));

  if (HPCTRACE_HDR_FLAGS_GET_BIT(flags, HPCTRACE_HDR_FLAGS_DATA_CENTRIC_BIT_POS)) {
    HPCFMT_ThrowIfError(hpcfmt_int4_fread(&(x->metricId), fs));
  }
  else {
    x->metricId = HPCTRACE_FMT_MetricId_NULL;
  }

  return HPCFMT_OK;
}


// Append the trace record to the outbuf.
// Returns: HPCFMT_OK on success, else HPCFMT_ERR.
int
hpctrace_fmt_datum_outbuf(hpctrace_fmt_datum_t* x, hpctrace_hdr_flags_t flags,
			  hpcio_outbuf_t* outbuf)
{
  const int bufSZ = sizeof(hpctrace_fmt_datum_t);
  unsigned char buf[bufSZ];
  int shift, k;

  k = 0;

  uint64_t comp = x->comp;
  for (shift = 56; shift >= 0; shift -= 8) {
    buf[k] = (comp >> shift) & 0xff;
    k++;
  }

  uint32_t cpId = x->cpId;
  for (shift = 24; shift >= 0; shift -= 8) {
    buf[k] = (cpId >> shift) & 0xff;
    k++;
  }

  if (HPCTRACE_HDR_FLAGS_GET_BIT(flags, HPCTRACE_HDR_FLAGS_DATA_CENTRIC_BIT_POS)) {
    uint32_t metricId = x->metricId;
    for (shift = 24; shift >= 0; shift -= 8) {
      buf[k] = (metricId >> shift) & 0xff;
      k++;
    }
  }

  if (hpcio_outbuf_write(outbuf, buf, k) != k) {
    return HPCFMT_ERR;
  }

  return HPCFMT_OK;
}


int
hpctrace_fmt_datum_fwrite(hpctrace_fmt_datum_t* x, hpctrace_hdr_flags_t flags,
			  FILE* outfs)
{
  HPCFMT_ThrowIfError(hpcfmt_int8_fwrite(x->comp, outfs));
  HPCFMT_ThrowIfError(hpcfmt_int4_fwrite(x->cpId, outfs));
  if (HPCTRACE_HDR_FLAGS_GET_BIT(flags, HPCTRACE_HDR_FLAGS_DATA_CENTRIC_BIT_POS)) {
    HPCFMT_ThrowIfError(hpcfmt_int4_fwrite(x->metricId, outfs));
  }

  return HPCFMT_OK;
}


int
hpctrace_fmt_datum_fprint(hpctrace_fmt_datum_t* x, hpctrace_hdr_flags_t flags,
			  FILE* fs)
{
  fprintf(fs, "(%llu, %u", HPCTRACE_FMT_GET_TIME(x->comp), x->cpId);
  if (HPCTRACE_HDR_FLAGS_GET_BIT(flags, HPCTRACE_HDR_FLAGS_LCA_RECORDED_BIT_POS)) {
    fprintf(fs, ", %llu",  HPCTRACE_FMT_GET_DLCA(x->comp));
  }
  if (HPCTRACE_HDR_FLAGS_GET_BIT(flags, HPCTRACE_HDR_FLAGS_DATA_CENTRIC_BIT_POS)) {
    fprintf(fs, ", %u",  x->metricId);
  }
  fputs(")\n", fs);
  return HPCFMT_OK;
}


//***************************************************************************
// hpcprof-metricdb (located here for now)
//***************************************************************************

//***************************************************************************
// [hpcprof-metricdb] hdr
//***************************************************************************

int
hpcmetricDB_fmt_hdr_fread(hpcmetricDB_fmt_hdr_t* hdr, FILE* infs)
{
  char tag[HPCMETRICDB_FMT_MagicLen + 1];
  char version[HPCMETRICDB_FMT_VersionLen + 1];
  char endian[HPCMETRICDB_FMT_EndianLen + 1];

  int nr = fread(tag, 1, HPCMETRICDB_FMT_MagicLen, infs);
  tag[HPCMETRICDB_FMT_MagicLen] = '\0';

  if (nr != HPCMETRICDB_FMT_MagicLen) {
    return HPCFMT_ERR;
  }
  if (strcmp(tag, HPCMETRICDB_FMT_Magic) != 0) {
    return HPCFMT_ERR;
  }

  nr = fread(&version, 1, HPCMETRICDB_FMT_VersionLen, infs);
  version[HPCMETRICDB_FMT_VersionLen] = '\0';
  if (nr != HPCMETRICDB_FMT_VersionLen) {
    return HPCFMT_ERR;
  }
  hdr->version = atof(hdr->versionStr);

  nr = fread(&endian, 1, HPCMETRICDB_FMT_EndianLen, infs);
  if (nr != HPCMETRICDB_FMT_EndianLen) {
    return HPCFMT_ERR;
  }

  HPCFMT_ThrowIfError(hpcfmt_int4_fread(&(hdr->numNodes), infs));
  HPCFMT_ThrowIfError(hpcfmt_int4_fread(&(hdr->numMetrics), infs));

  return HPCFMT_OK;
}


int
hpcmetricDB_fmt_hdr_fwrite(hpcmetricDB_fmt_hdr_t* hdr, FILE* outfs)
{
  int nw;

  nw = fwrite(HPCMETRICDB_FMT_Magic,   1, HPCMETRICDB_FMT_MagicLen, outfs);
  if (nw != HPCTRACE_FMT_MagicLen) return HPCFMT_ERR;

  nw = fwrite(HPCMETRICDB_FMT_Version, 1, HPCMETRICDB_FMT_VersionLen, outfs);
  if (nw != HPCMETRICDB_FMT_VersionLen) return HPCFMT_ERR;

  nw = fwrite(HPCMETRICDB_FMT_Endian,  1, HPCMETRICDB_FMT_EndianLen, outfs);
  if (nw != HPCMETRICDB_FMT_EndianLen) return HPCFMT_ERR;

  HPCFMT_ThrowIfError(hpcfmt_int4_fwrite(hdr->numNodes, outfs));
  HPCFMT_ThrowIfError(hpcfmt_int4_fwrite(hdr->numMetrics, outfs));

  return HPCFMT_OK;
}


int
hpcmetricDB_fmt_hdr_fprint(hpcmetricDB_fmt_hdr_t* hdr, FILE* outfs)
{
  fprintf(outfs, "%s\n", HPCMETRICDB_FMT_Magic);
  fprintf(outfs, "[hdr:...]\n");

  fprintf(outfs, "(num-nodes:   %u)\n", hdr->numNodes);
  fprintf(outfs, "(num-metrics: %u)\n", hdr->numMetrics);

  return HPCFMT_OK;
}

