clc
clear


load('inputs_outputs_sets.mat')
input_domain = double([1,1,1;
    1,1,-1;
    1,-1,1;
    1,-1,-1;
    -1,1,1;
    -1,1,-1;
    -1,-1,1;
    -1,-1,-1]);

for num = 1:length(all_out_sets)
   
    out_sets = all_out_sets(num,:);
    unsafe_domain = squeeze(all_unsafe_domain(num,:,:));
    unsafe_sets = all_unsafe_sets{1,num};
    
    f1 = figure('visible','off');
    sgtitle(['Instance ', num2str(num)])
    f1.Position = [100 400 900 400];
    
    subplot(1,2,1)
    title('Input domain & Exact unsafe subspace')  
    hold on 
    for n = 1:length(unsafe_sets)
        aset_vertices = unsafe_sets{1,n};
        polyh_output = Polyhedron('V', aset_vertices);
        polyh_output.plot('edgealpha',0.2,'alpha',0.5)
        xlim([-1,1])
        ylim([-1,1])
        zlim([-1,1])
        xlabel('x_1', 'FontSize', 15)
        ylabel('x_2', 'FontSize', 15)
        zlabel('x_3', 'FontSize', 15)
    end
    input_domain_poly = Polyhedron('V', input_domain);
    input_domain_poly.plot('edgealpha',1.0,'alpha',0.2,'color','b')
    hold off
    
    subplot(1,2,2)
    title('Exact output reachable domain & Unsafe domain')  
    hold on 
    for n = 1:length(out_sets)
        aset_vertices = out_sets{1,n};
        polyh_output = Polyhedron('V', aset_vertices);
        polyh_output.plot('edgealpha',0.0,'alpha',1, 'color', 'b')
        xlabel('y_1', 'FontSize', 15)
        ylabel('y_2', 'FontSize', 15)
    end
    unsafe_domain_poly = Polyhedron('V', double(unsafe_domain));
    unsafe_domain_poly.plot('edgealpha',0.5,'alpha',0.5, 'color', 'r')
    hold off

    if num <=9
        print(f1, ['images/reach00', num2str(num),'.png'], '-dpng')
    elseif num<=99
        print(f1, ['images/reach0', num2str(num),'.png'], '-dpng')
    end
    
end



