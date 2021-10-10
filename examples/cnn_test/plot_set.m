clc
clear

f = figure();
hold on
load('vertices.mat')
vertices = double(vertices);
S = Polyhedron('V', vertices);
S.plot()

vertices0 = double(vertices0);
S0 = Polyhedron('V', vertices0);
S0.plot('color','b')
hold off


f = figure();
hold on
load('vertices.mat')
vertices = double(vertices);
S = Polyhedron('V', vertices);
S.plot()

vertices1 = double(vertices1);
S1 = Polyhedron('V', vertices1);
S1.plot('color','y')
hold off
