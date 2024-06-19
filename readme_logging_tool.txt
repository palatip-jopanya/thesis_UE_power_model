A guide for logging tool
# this tool is created for converting & combining power mesurement log and CMX500 log into CSV format.

To run the tool:
1. Collect both power measurement (HV Monsoon) and CMX500 logs using script (see scripts in folder CMX500_sweep_script)
2. Run python file main.py in any python IDE.
3. A first pop-up window for file selection: select CMX500 log (log from script execution in CMX500 i.e. .PCT)
4. A second pop-up window for file selection: select power mesurement log (log from HV monsoon in CSV format)
5. If there isn't any error the program will output the converted file that is located in the same folder as the main.py is located.

Setting:
There is a possibility for setting a timing offset i.e. to synchronize the actual starting time in the power measurement log with CMX500. By editing the variable "offsetpwrtime" in the main.py, the default is 0 second.
By setting "offsetpwrtime" as 1.5 means the program will start look at the power measurement log starting from 1.5 s.
