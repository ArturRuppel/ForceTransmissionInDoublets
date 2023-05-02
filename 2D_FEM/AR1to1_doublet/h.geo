// Gmsh project created on Tue Sep  8 00:08:27 2020
// Gmsh project AR1to1
// Inputs

CDL = 45;
Linklength = 1;
gridsize = Linklength;

Point(1) = {-CDL/2,-CDL/2,0,gridsize};
Point(2) = {CDL/2,-CDL/2,0,gridsize};
Point(3) = {CDL/2,CDL/2,0,gridsize};
Point(4) = {-CDL/2,CDL/2,0,gridsize};



Line(7) = {1,2};
Line(8) = {2,3};
Line(9) = {3,4};
Line(10) = {4,1};

Line Loop(11) = {7,8,9,10};


Plane Surface(12) = 11;

