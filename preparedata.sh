#!/bin/bash

dir=ft-drv-grammar
rm -rf data
mkdir data
for i in split-extract-data/*; do
    cp -f $i/`basename $i`"_M"*/*.{drv,sel,snk}.csv ./data
done
