#!/bin/bash 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 5 16 16 3 24 > System_5/MyRunInfo/Run_16.txt & 
wait
python3 deepDMD.py '/cpu:0' 5 17 16 4 24 > System_5/MyRunInfo/Run_17.txt & 
wait
python3 deepDMD.py '/cpu:0' 5 18 20 3 30 > System_5/MyRunInfo/Run_18.txt & 
wait
python3 deepDMD.py '/cpu:0' 5 19 20 4 30 > System_5/MyRunInfo/Run_19.txt & 
wait
python3 deepDMD.py '/cpu:0' 5 20 24 3 36 > System_5/MyRunInfo/Run_20.txt & 
wait
python3 deepDMD.py '/cpu:0' 5 21 24 4 36 > System_5/MyRunInfo/Run_21.txt & 
wait
echo "All sessions are complete" 
echo "=======================================================" 
