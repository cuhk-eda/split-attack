#!/bin/bash

dir=ft-drv-grammar
rm -rf data-${dir}
mkdir data-${dir}
for i in split-extract-${dir}/*; do
    cp -f $i/`basename $i`"_M"*/*.{drv,sel,snk}.csv ./data-${dir}/
done

