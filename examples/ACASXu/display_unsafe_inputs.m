clear
clc

if ~isfolder('tbxmanager')
    mkdir 'tbxmanager' 
end

addpath('../../class')
run install_mpt3.m

load('logs/unsafe_inputs_p2_N12.mat')

if length(unsafe_inputs)==0
   return
end

for n = 1:length(unsafe_inputs)
    aset = unsafe_inputs(n);
    vs = aset{1};
    for i = 1:5
        vs(:,i) = vs(:,i)*rangex(i)+meanx(i);
    end
    polys1(n) = Polyhedron('V',vs(:,[1,2,3]));
    polys2(n) = Polyhedron('V',vs(:,[1,4,5]));
end

fig = figure;
subplot(1,2,1)
polys1.plot('edgealpha',0.1,'alpha',1.0)
xlabel('\rho')
ylabel('\theta')
zlabel('\psi')
subplot(1,2,2)
polys2.plot('edgealpha',0.1,'alpha',1.0)
xlabel('\rho')
ylabel('v_{own}')
zlabel('v_{int}')
t = suptitle('Figure 4: Complete input sets that lead to the property 2 violation on the network N12');
t.FontSize = 12;
set(fig,'Position',[100, 100, 800, 400])
saveas(fig,'Figure4.png')
