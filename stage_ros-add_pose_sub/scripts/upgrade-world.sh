#!/bin/bash

USAGE="USAGE: upgrade-world.sh in.world out.world"

if [ $# != 2 ]; then
  echo $USAGE
  exit 1
fi

cat < $1 | sed 's/size3/size/' | sed 's/origin3/origin/' | sed 's/.*interval_real.*//' | sed 's/.*range_min.*//' | sed 's/.*center.*//' | sed 's/.*gui_movemask.*//' | sed 's/\(.*pose *\[\)\([^ ]*\) *\([^ ]*\) *\([^ ]*\) *\(\].*\)/\1 \2 \3 0 \4 \5/' > $2
