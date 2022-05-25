#!/bin/bash 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 5 10 16 3 24 > System_5/MyRunInfo/Run_10.txt & 
wait
python3 deepDMD.py '/cpu:0' 5 11 16 4 24 > System_5/MyRunInfo/Run_11.txt & 
wait
python3 deepDMD.py '/cpu:0' 5 12 20 3 30 > System_5/MyRunInfo/Run_12.txt & 
wait
python3 deepDMD.py '/cpu:0' 5 13 20 4 30 > System_5/MyRunInfo/Run_13.txt & 
wait
python3 deepDMD.py '/cpu:0' 5 14 24 3 36 > System_5/MyRunInfo/Run_14.txt & 
wait
python3 deepDMD.py '/cpu:0' 5 15 24 4 36 > System_5/MyRunInfo/Run_15.txt & 
wait
echo "All sessions are complete" 
echo "=======================================================" 
