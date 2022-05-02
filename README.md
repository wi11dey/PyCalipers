# PyCalipers
Automatically detect layer orientation and thicknesses with accurate standard deviations in images. Originally used to process Scanning Electron Microscope images, but general enough to process arbitrary layers, even geological sediment, for example. Can also be imported as a `calipers` library.

## Gaussian fit and standard deviations

![CS_normalView_03.tif](https://raw.githubusercontent.com/wi11dey/PyCalipers/master/CS_normalView_03_fit.png)

## Orientation detection

<img src="https://raw.githubusercontent.com/wi11dey/PyCalipers/master/image7.png" width="500" /><img src="https://raw.githubusercontent.com/wi11dey/PyCalipers/master/image8.png" width="500" /><img src="https://raw.githubusercontent.com/wi11dey/PyCalipers/master/image11.png" width="500" />

## Example
```
$ calipers.py CS_normalView_03.tif -o CS_normalView_03.csv
Reading...
Cropping...
Detecting orientation...
Rotating by 1.16407872°...
Differentiating...
Fitting Gaussian distributions...
Scaling...

$ cat CS_normalView_03.csv
Layer #,Thickness,Standard deviation,Units
1,106.890093,42.681708, nm
2,174.410448,42.713747, nm
3,107.248487,31.913159, nm
4,37.172840,46.892665, nm
5,62.560009,49.661766, nm
6,105.235871,36.444696, nm
7,159.979666,44.454964, nm
8,116.023772,43.562002, nm
9,369.716397,46.699883, nm
10,116.645687,43.981174, nm
11,155.810149,41.191307, nm
12,110.719272,40.908501, nm
13,80.611800,27.554137, nm
14,52.306222,14.994458, nm
15,84.130643,30.202688, nm
16,171.701936,44.438595, nm
17,111.500654,41.684820, nm
18,178.402873,40.577570, nm
19,144.529521,33.747483, nm
20,28.615011,10.351548, nm
21,112.093959,32.713880, nm
22,184.488292,46.618902, nm
23,113.293174,40.413320, nm
24,71.385392,25.874862, nm
25,255.580606,20.016292, nm
26,91.743626,24.327549, nm
27,129.679163,29.091099, nm
28,218.338424,29.518431, nm
29,130.067721,28.666897, nm
30,217.953899,27.983413, nm
31,131.219414,27.302246, nm
32,205.978617,25.977732, nm
33,125.231194,20.332181, nm
34,35.635510,13.453500, nm
35,122.200944,19.432452, nm
36,204.321501,24.097702, nm
37,130.492771,23.434461, nm
38,213.594431,22.834548, nm
39,114.034742,22.350233, nm
40,166.448330,21.778308, nm
41,102.911302,21.203220, nm
42,170.224846,20.902043, nm
...
```
