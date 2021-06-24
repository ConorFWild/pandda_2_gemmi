#!/bin/sh
version="Time-stamp: <2020-12-07 11:21:37 vonrhein>"
contact="buster-develop"
#
# Copyright  2020 by Global Phasing Limited
#
#           All rights reserved.
#
#           This software is proprietary to and embodies the confidential
#           technology of Global Phasing Limited (GPhL). Possession, use,
#           duplication or dissemination of the software is authorised
#           only pursuant to a valid written licence from GPhL.
#
# Author    (2020) Clemens Vonrhein <vonrhein@GlobalPhasing.com>
#
# Contact   $contact@GlobalPhasing.com
#
#----------------------------------------------------------------------
#              BEGIN OF USER INPUT
#----------------------------------------------------------------------

# some generic and useful settings:
cdir=`pwd`
tmp=/tmp/`whoami`-`date +%s`_$$
exes=""
files=""
iverb=0

#----------------------------------------------------------------------
#               END OF USER INPUT
#----------------------------------------------------------------------
echo " " >&2
echo " ============================================================================ " >&2
echo " " >&2
echo " Copyright (C) 2020 by Global Phasing Limited" >&2
echo " " >&2
echo "           All rights reserved." >&2
echo " " >&2
echo "           This software is proprietary to and embodies the confidential" >&2
echo "           technology of Global Phasing Limited (GPhL). Possession, use," >&2
echo "           duplication or dissemination of the software is authorised" >&2
echo "           only pursuant to a valid written licence from GPhL." >&2
echo " " >&2
echo " ---------------------------------------------------------------------------- " >&2
echo " " >&2
echo " Author:   (2020) Clemens Vonrhein" >&2
echo " " >&2
echo " Contact:  $contact@GlobalPhasing.com" >&2
echo " " >&2
ShortVersion=`echo $version | cut -f2- -d':' | sed "s/ [a-z0-9][a-z0-9][a-z0-9]*>/>/g"`
echo " Program:  `basename $0`   version ${ShortVersion} " >&2
echo " " >&2
echo " ============================================================================ " >&2
echo " " >&2
#----------------------------------------------------------------------
#               BEGIN OF SCRIPT
#----------------------------------------------------------------------

# --------- functions
error () {
  echo " "
  [ $# -ne 0 ] && echo " ERROR: $@" || echo " ERROR: see above"
  echo " "
  exit 1
}
warning () {
  echo " "
  [ $# -ne 0 ] && echo " WARNING: $@" || echo " WARNING: see above"
  echo " "
}
note () {
  if [ $# -gt 0 ]; then
    echo " "
    echo " NOTE: $@"
    echo " "
  fi
}
usage () {
  echo " "
  echo " USAGE: $0 [-h] [-v] -pdb <PDB> -map <MAP> -mtz <MTZ> -cif <CIF> -out <dir> [-cut <cutoff>] [-vol <Volume>]"
  echo " "
  echo "        -h                    : show help"
  echo " "
  echo "        -v                    : increase verbosity (default = $iverb)"
  echo " "
  echo "        -pdb <PDB>            : (positioned) apo PDB file"
  echo " "
  echo "        -mtz <MTZ>            : corresponding MTZ file"
  echo " "
  echo "        -cif <CIF>            : compound CIF"
  echo " "
  echo "        -map <MAP>            : PanDDA event map (full asymmetric unit)"
  echo " "
  echo "        -out <dir>            : output directory (default = $out)"
  echo " "
  echo "        -cut <cutoff>         : clustering cutoff (default = $cut)"
  echo " "
  echo "        -vol <colume>         : minimum cluster volume (default = $vol)"
  echo " "
}
chkarg () {
  __a=$1
  __m=$2
  __n=$3
  if [ $__n -lt $__m ]; then
    usage
    error "not enough arguments for command-line flag \"$__a\""
  fi
}

# --------- process command-line
vars=""
pdb=""
mtz=""
map=""
out="."
cut=1.0
vol=20
while [ $# -gt 0 ]
do
  case $1 in
    -v) iverb=`expr $iverb + 1`;;
    -h) usage; exit 0;;
    -p*) pdb=$2;shift;;
    -mtz) mtz=$2;shift;;
    -map) map=$2;shift;;
    -out) out=$2;shift;;
    -cif) cif=$2;shift;;
    -cut) cut=$2;shift;;
    -vol) col=$2;shift;;
     *) usage;error "unknown argument \"$1\"";;
  esac
  shift
done

# --------- checks
for var in $vars pdb mtz cif map out
do
  eval "val=\$$var"
  [ "X$val" = "X" ] && usage && error "variable \"$var\" not set"
done
for exe in $exes ana_pdbmaps rhofit
do
  type $exe >/dev/null 2>&1
  [ $? -ne 0 ] && error "executable \"$exe\" not found (in PATH)"
done
for f in $files $pdb $mtz $map $cif
do
  [ ! -f $f ] && error "file \"$f\" not found"
  [ ! -r $f ] && error "file \"$f\" not readable"
done

# --------- start doing something

for f in pdb mtz map cif
do
  eval "ff=\$$f"
  case $ff in
    /*) true;;
     *) eval "$f=\"\$cdir/\$ff\"";;
  esac
  eval "ff=\$$f"
  echo " $f = $ff"
done
echo " "

printf " running ANA_PDBMAPS to turn event map into cluster (for fitting) ... "
t0=`date +%s`
ana_pdbmaps -single_map -cutabs <<e > $out/ana_pdbmaps.log 2>&1
$pdb
$map
5.0
$cut
$vol
$out/ana_pdbmaps.pdb
e
[ $? -ne 0 ] && error "see $out/ana_pdbmaps.log"
t1=`date +%s`
echo "done (`expr $t1 - $t0` sec)"
[ ! -f $out/ana_pdbmaps.pdb ] && error "no cluster(s) found - see $out/ana_pdbmaps.log"

printf " creating (potentially) separate cluster(s) and inex file ... "
t0=`date +%s`
awk -v o=$out '/^REMARK/{nr++;lr[nr]=$0;next}
     /^CRYST1/{nr++;lr[nr]=$0;next}
     /^MODEL/{
       out=o "/" sprintf("ana_pdbmaps_cluster_%3.3d",$2)
       getline
       print > out
       for(i=1;i<=nr;i++) print lr[i] >> out
       next
     }
     {print>>out}' $out/ana_pdbmaps.pdb
ls -1 $out/ana_pdbmaps_cluster_[0-9][0-9][0-9] | sed "s%.*/%%g" > $out/ana_pdbmaps_cluster.lis
t1=`date +%s`
echo "done (`expr $t1 - $t0` sec)"

(
  cd $out && \
  printf " running Rhofit ... " && \
  t0=`date +%s` && \
  rhofit -allclusters -m $mtz -l $cif -p $pdb -d rhofit -C ana_pdbmaps_cluster.lis > rhofit.lis 2>&1
  [ $? -ne 0 ] && error "see $out/rhofit.lis"
  cd $cdir
  t1=`date +%s`
  echo "done (`expr $t1 - $t0` sec)"
  echo " "
  ls -l $out/rhofit/*.pdb | awk '{print "    ",$0}'
  echo " "
  awk '/ LigProt /{ido=1}{if(NF>0&&ido==1)print}' $out/rhofit.lis
  echo " "
  cat <<e

 You can visualise results via

   cd $out/rhofit
   visualise-rhofit-coot
e
)

# --------- finish
rm -fr ${tmp}*
echo " " >&2
echo " ... normal termination ... " >&2
echo " " >&2
exit 0
#----------------------------------------------------------------------
#               END OF SCRIPT
#----------------------------------------------------------------------
