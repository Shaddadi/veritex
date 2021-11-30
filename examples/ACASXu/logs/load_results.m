clc
clear
% 
% %%%%%%%%%%%%%% our method %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% folder1 = './logs_lr0.001_epochs200';
% files = dir(folder1);
% 
% all_accu = [];
% for num = 3:length(files)
%     filename = files(num).name;
%     try
%         load([folder1, '/', filename, '/all_test_accuracy.mat'])
%         accu = all_test_accuracy(end);
%         all_accu(end+1) = accu;
%     catch
%         continue
%     end
% end
% min(all_accu)
% mean(all_accu)
% max(all_accu)


folder1 = './logs_lr0.001_epochs200';
files = dir(folder1);

fig1 = figure();
hold on
traces_filled = [];
for num = 3:length(files)
    filename = files(num).name;
    try
        load([folder1, '/', filename, '/all_test_accuracy.mat'])
        
        if length(all_test_accuracy)<25 % 25 is the longest trace
            tmp_trace = ones(1,25-length(all_test_accuracy))*all_test_accuracy(end);
            tmp_trace = [all_test_accuracy, tmp_trace];
            traces_filled = [traces_filled; tmp_trace];
        else
            traces_filled = [traces_filled; all_test_accuracy];
        end
        
    catch
        continue
    end
end

ubs = max(traces_filled,[],1);
lbs = min(traces_filled,[],1);
mbs = mean(traces_filled,1);
x = 1:25;
fill([x fliplr(x)], [lbs fliplr(ubs)],[0.9,0.9,0.9])

for num = 3:length(files)
    filename = files(num).name;
    try
        load([folder1, '/', filename, '/all_test_accuracy.mat'])
        p = plot(all_test_accuracy,'-','color',[0.7 0.7 0.7],'LineWidth',0.5);
        
        if length(all_test_accuracy)<25 % 25 is the longest trace
            tmp_trace = ones(1,25-length(all_test_accuracy))*all_test_accuracy(end);
            tmp_trace = [all_test_accuracy, tmp_trace];
            traces_filled = [traces_filled; tmp_trace];
        else
            traces_filled = [traces_filled; all_test_accuracy];
        end
        
    catch
        continue
    end
end
p1 = plot(ubs,'-','color',[0.5 0.0 0.5],'LineWidth',1.5);
p2 = plot(lbs,'-','color',[0.0 0.5 0.0],'LineWidth',1.5);
p3 = plot(mbs,'-','color',[0.0 0.0 0.5],'LineWidth',1.5);
set(gca,'FontSize',15)
xlabel('Iteration of repairing','FontSize',15)
% ylabel('Accuracy(%)','FontSize',15)
% title({'Neural Network Repair',['35','/',num2str(num-2),' success'],'Learning Rate 0.001 and Epochs 200'})
legend([p1(1),p2(1),p3(1),p(1)],'Upper Bound','Lower Bound','Average', 'Instance');
ylim([0.984, 1.0])
hold off


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% art %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load('art_accuracy_time.mat')
% all_accu = [];
% for num = 1:length(results_refine)
%     accuracy = results_refine{num,1}(end);
%     all_accu(end+1) = accuracy;
% end
% min(all_accu)
% mean(all_accu)
% max(all_accu)
% 
% all_accu = [];
% for num = 1:length(results_no_refine)
%     accuracy = results_no_refine{num,1}(end);
%     all_accu(end+1) = accuracy;
% end
% min(all_accu)
% mean(all_accu)
% max(all_accu)

load('art_accuracy_time.mat')

traces_filled = [];
for num = 1:length(results_refine)
    accuracys = results_refine{num,1};
    traces_filled = [traces_filled; accuracys];
end

fig2 = figure();
hold on

ubs = max(traces_filled,[],1);
lbs = min(traces_filled,[],1);
mbs = mean(traces_filled,1);
x = 1:26;
fill([x fliplr(x)], [lbs fliplr(ubs)],[0.9,0.9,0.9])

low_accuracys = [];
for num = 1:length(results_refine)
    accuracys = results_refine{num,1};
    p = plot(accuracys,'-','color',[0.7 0.7 0.7],'LineWidth',0.5);
end

p1 = plot(ubs,'-','color',[0.5 0.0 0.5],'LineWidth',1.5);
p2 = plot(lbs,'-','color',[0.0 0.5 0.0],'LineWidth',1.5);
p3 = plot(mbs,'-','color',[0.0 0.0 0.5],'LineWidth',1.5);
set(gca,'FontSize',15)
xlabel('Iteration of repairing','FontSize',15)
ylabel('Accuracy(%)','FontSize',15)
legend([p1(1),p2(1),p3(1),p(1)],'Upper Bound','Lower Bound','Average', 'Instance');
ylim([0.1,1.0])
hold off 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fig3 = figure();
% hold on
% for num = 1:length(results_no_refine)
%     accuracys = results_no_refine{num,1};
%     plot(accuracys,'-s','color',[0.7 0.0 0.0],'MarkerSize',4)
%     plot(length(accuracys),accuracys(end),'-bs','MarkerSize',4)
% end
% xlabel('Iteration of repairing')
% ylabel('Accuracy(%)')
% hold off 