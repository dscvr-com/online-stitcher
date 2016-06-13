set terminal png size 20000,2000 linewidth 2.0
set output "transforms.png"

set pointsize 0.7
set style line 6 lc rgb "#f0b000"
set style line 15 lc rgb "#f0b000"
set style line 24 lc rgb "#f0b000"
set style line 33 lc rgb "#f0b000"
set style line 42 lc rgb "#f0b000"
set style line 51 lc rgb "#f0b000"
set style line 60 lc rgb "#f0b000"
set style increment user

set grid xtics ytics

set key top left

set title 'IAM 360 Stabilization test'
set xlabel 't'
set ylabel 'radians/unitless'

# IMPORT-DATA transforms transforms.txt 
## MULTIPLOT(transforms) SELECT t AS x, val AS y, type AS MULTIPLOT
## FROM transforms
## WHERE type = 'sensor-abs-x' OR type = 'sensor-int-x' OR type = 'sensor-diff-y' OR type = 'estimated-diff-y' 
## GROUP BY MULTIPLOT,t ORDER BY MULTIPLOT,t
plot \
    'transforms-data.txt' index 0 title "transforms=estimated-diff-y" with linespoints, \
    'transforms-data.txt' index 1 title "transforms=sensor-abs-x" with linespoints, \
    'transforms-data.txt' index 2 title "transforms=sensor-diff-y" with linespoints, \
    'transforms-data.txt' index 3 title "transforms=sensor-int-x" with linespoints
