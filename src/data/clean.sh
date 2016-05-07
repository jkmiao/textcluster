#!/bin/bash


sed '/^$/d' ./input_data.txt > cleaned_data.txt
sed -i 's/^[ \t ]*//g' cleaned_data.txt
sed -i 's/[ \t]$//g' cleaned_data.txt
