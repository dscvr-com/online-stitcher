This directory contains some auxiallry files that can be used to plot certain values. 

For this, the values are simply printed to stdout and then collected in a file. 

Then, the file and the plot-file are processed with ```sqlplot-tools``` and finally plotted using ```gnuplot```.

Example: ```sp-process transforms.plot && gnuplot transforms.plot```.
