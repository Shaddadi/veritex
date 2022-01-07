clc
clear
% 



% %%%%%%%%%%%%%%%%%%%%%%%%%%% minimal repair %%%%%%%%%%%%%%%%%%%%%
% all_lr = [0.01,0.001];
% all_alpha_beta = [[0.2, 0.8];[0.5, 0.5]; [0.8,0.2]];
% 
% all_accu = [];
% all_times = [];
% for nn_i = 1:5
%     for nn_j = 1:9
%         accu_nnet = [];
%         for lr_id = 1:length(all_lr)
%             lr = all_lr(lr_id);
%             for ab_id = 1:length(all_alpha_beta)
%                 alpha = all_alpha_beta(ab_id, 1);
%                 beta = all_alpha_beta(ab_id, 2);
%                 
%                 dir1 = 'logs_lrxxx_epochs200_alpha_beta/';
%                 dir2 = ['nnet',num2str(nn_i),num2str(nn_j),'_lr',num2str(lr),...
%                     '_epochs200','_alpha',num2str(alpha),'_beta',num2str(beta),'/all_test_accuracy.mat'];
%                 
%                 try
%                     load([dir1,dir2])
%                     accu = all_test_accuracy(end);
%                     accu_nnet(end+1) = accu;
%                 catch
%                     continue
%                 end
%             end
%         end
%         if ~isempty(accu_nnet)
%             [max_accu, max_id] = max(accu_nnet);
%             all_accu(end+1) = max_accu;
%         end
%     end
% end
% 
% min(all_accu)
% mean(all_accu)
% max(all_accu)

%%%%%%%%%%%%%%%%%%%%%%%%%% time %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%% linear regions%%%%%%%%%%%%%%%%%%%%%%
% load('all_our_minimal_lregions.mat')
% all_lr = [];
% for num = 1:length(all_our_lregions)
%     ori = all_ori_lregions(num);
%     val = all_our_lregions(num);
% %     double(abs(ori-val))/double(ori)
%     all_lr(end+1) = double(abs(ori-val))/double(ori);
% %     fprintf('%.3f\n',double(abs(ori-val))/double(ori))
% end
% min(all_lr)
% mean(all_lr)
% max(all_lr)



% load('all_linear_regions.mat')

% all_lr = [];
% for num = 1:length(all_our_lregions)
%     ori = all_ori_lregions(num);
%     val = all_our_lregions(num);
% %     double(abs(ori-val))/double(ori)
%     all_lr(end+1) = double(abs(ori-val))/double(ori);
% %     fprintf('%.3f\n',double(abs(ori-val))/double(ori))
% end
% min(all_lr)
% mean(all_lr)
% max(all_lr)


% all_lr = [];
% for num = 1:length(all_art_lregion_refines)
%     ori = all_ori_lregions(num);
%     val = all_art_lregion_refines(num);
% %     double(abs(ori-val))/double(ori)
%     all_lr(end+1) = double(abs(ori-val))/double(ori);
%     fprintf('%.3f\n',double(abs(ori-val))/double(ori))
% end
% min(all_lr)
% mean(all_lr)
% max(all_lr)

% all_lr = [];
% for num = 1:length(all_art_lregion_no_refines)
%     ori = all_ori_lregions(num);
%     val = all_art_lregion_no_refines(num);
% %     double(abs(ori-val))/double(ori)
%     all_lr(end+1) = double(abs(ori-val))/double(ori);
% %     fprintf('%.3f\n',double(abs(ori-val))/double(ori))
% end
% min(all_lr)
% mean(all_lr)
% max(all_lr)

% %%%%%%%%%%%%%%%%%%%%% weights deviation%%%%%%%%%%%%%%%%%%
load('weights_deviation.mat')

all_pd = [];
for num = 1:length(all_our_deviation)
    val = all_our_deviation(num);
    all_pd(end+1) = val;
%     fprintf('%.3f\n',val);
end
min(all_pd)
mean(all_pd)
max(all_pd)


all_pd = [];
for num = 1:length(all_minimal_repair_deviation)
    val = all_minimal_repair_deviation(num);
    all_pd(end+1) = val;
%     fprintf('%.3f\n',val);
end
min(all_pd)
mean(all_pd)
max(all_pd)

% 
all_pd = [];
for num = 1:length(all_art_refine_dev)
    val = all_art_refine_dev(num);
    all_pd(end+1) = val;
%     fprintf('%.2f\n',val)
end
% load('all_our_minimal_lregions.mat')
% all_lr = [];
% for num = 1:length(all_our_lregions)
%     ori = all_ori_lregions(num);
%     val = all_our_lregions(num);
% %     double(abs(ori-val))/double(ori)
%     all_lr(end+1) = double(abs(ori-val))/double(ori);
% %     fprintf('%.3f\n',double(abs(ori-val))/double(ori))
% end
% min(all_lr)
% mean(all_lr)
% max(all_lr)

% min(all_pd)
% mean(all_pd)
% max(all_pd)
% 
% all_pd = [];
% for num = 1:length(all_art_no_refine_dev)
%     val = all_art_no_refine_dev(num);
%     all_pd(end+1) = val;
% %     fprintf('%.2f\n',val)
% end
% min(all_pd)
% mean(all_pd)
% max(all_pd)
%%%%%%%%%%%%%% our method %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% folder1 = './logs_lr0.001_epochs200';
% files = dir(folder1);
% 
% all_accu = [];
% all_time = [];
% for num = 3:length(files)
%     filename = files(num).name;
%     try
%         load([folder1, '/', filename, '/all_test_accuracy.mat'])
%         accu = all_test_accuracy(end);
%         all_accu(end+1) = accu;
%         if time<3000
%             all_time(end+1) = time;
%         end
%     catch
%         continue
%     end
% end
% % min(all_accu)
% % mean(all_accu)
% % max(all_accu)
% 
% min(all_time)
% mean(all_time)
% max(all_time)



% folder1 = './logs_lr0.001_epochs200';
% files = dir(folder1);
% 
% fig1 = figure();
% hold on
% traces_filled = [];
% for num = 3:length(files)
%     filename = files(num).name;
%     try
%         load([folder1, '/', filename, '/all_test_accuracy.mat'])
%         % load('weights_deviation.mat')
% 
% all_pd = [];
% for num = 1:length(all_our_deviation)
%     val = all_our_deviation(num);
%     all_pd(end+1) = val;
%     fprintf('%.3f\n',val)
% end
% min(all_pd)
% mean(all_pd)
% max(all_pd)

%         if length(all_test_accuracy)<25 % 25 is the longest trace
%             tmp_trace = ones(1,25-length(all_test_accuracy))*all_test_accuracy(end);
%             tmp_trace = [all_test_accuracy, tmp_trace];
%             traces_filled = [traces_filled; tmp_trace];
%         else
%             traces_filled = [traces_filled; all_test_accuracy];
%         end
%         
%     catch
%         continue
%     end
% end
% 
% ubs = max(traces_filled,[],1);
% lbs = min(traces_filled,[],1);
% mbs = mean(traces_filled,1);
% x = 1:25;
% fill([x fliplr(x)], [lbs fliplr(ubs)],[0.9,0.9,0.9])
% 
% for num = 3:length(files)
%     filename = files(num).name;
%     try
%         load([folder1, '/', filename, '/all_test_accuracy.mat'])
%         p = plot(all_test_accuracy,'-','color',[0.7 0.7 0.7],'LineWidth',0.5);
%         
%         if length(all_test_accuracy)<25 % 25 is the longest trace
%             tmp_trace = ones(1,25-length(all_test_accuracy))*all_test_accuracy(end);
%             tmp_trace = [all_test_accuracy, tmp_trace];
%             traces_filled = [traces_filled; tmp_trace];
%         else
%             traces_filled = [traces_filled; all_test_accuracy];
%         end
%         
%     catch
%         continue
%     end
% end
% p1 = plot(ubs,'-','color',[0.5 0.0 0.5],'LineWidth',1.5);
% p2 = plot(lbs,'-','color',[0.0 0.5 0.0],'LineWidth',1.5);
% p3 = plot(mbs,'-','color',[0.0 0.0 0.5],'LineWidth',1.5);
% set(gca,'FontSize',15)
% xlabel('Iteration of repairing','FontSize',15)
% % ylabel('Accuracy(%)','FontSize',15)
% % title({'Neural Network Repair',['35','/',num2str(num-2),' success'],'Learning Rate 0.001 and Epochs 200'})
% legend([p1(1),p2(1),p3(1),p(1)],'Upper Bound','Lower Bound','Average', 'Instance');
% ylim([0.984, 1.0])
% hold off


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% art %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load('art_accuracy_time.mat')
% all_accu = [];
% all_time = [];
% for num = 1:length(results_refine)
%     accuracy = results_refine{num,1}(end);
% %     fprintf('%.2f%%\n',accuracy*100)
%     all_accu(end+1) = accuracy;
%     all_time(end+1) = results_refine{num,2}(end);
% end
% % min(all_accu)
% % mean(all_accu)
% % max(all_accu)
% all_time(1)
% all_time(2)
% min(all_time(3:end))
% mean(all_time(3:end))
% max(all_time(3:end))

% all_accu = [];
% all_time = [];
% for num = 1:length(results_no_refine)
%     accuracy = results_no_refine{num,1}(end);
% %     fprintf('%.2f%%\n',accuracy*100)
%     all_accu(end+1) = accuracy;
%     all_time(end+1) = results_no_refine{num,2}(end);
% end
% % min(all_accu)
% % mean(all_accu)
% % max(all_accu)
% 
% all_time(1)
% all_time(2)
% min(all_time(3:end))
% mean(all_time(3:end))
% max(all_time(3:end))

% load('art_accuracy_time.mat')
% 
% traces_filled = [];
% for num = 1:length(results_refine)
%     accuracys = results_refine{num,1};
%     traces_filled = [traces_filled; accuracys];
% end
% 
% fig2 = figure();
% hold on
% 
% ubs = max(traces_filled,[],1);
% lbs = min(traces_filled,[],1);
% mbs = mean(traces_filled,1);
% x = 1:26;
% fill([x fliplr(x)], [lbs fliplr(ubs)],[0.9,0.9,0.9])
% 
% low_accuracys = [];
% for num = 1:length(results_refine)
%     accuracys = results_refine{num,1};
%     p = plot(accuracys,'-','color',[0.7 0.7 0.7],'LineWidth',0.5);
% end
% 
% p1 = plot(ubs,'-','color',[0.5 0.0 0.5],'LineWidth',1.5);
% p2 = plot(lbs,'-','color',[0.0 0.5 0.0],'LineWidth',1.5);
% p3 = plot(mbs,'-','color',[0.0 0.0 0.5],'LineWidth',1.5);
% set(gca,'FontSize',15)
% xlabel('Iteration of repairing','FontSize',15)
% ylabel('Accuracy(%)','FontSize',15)
% legend([p1(1),p2(1),p3(1),p(1)],'Upper Bound','Lower Bound','Average', 'Instance');
% ylim([0.1,1.0])
% hold off 

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