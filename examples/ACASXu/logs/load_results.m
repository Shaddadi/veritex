
folder1 = './logs_lr0.001_epochs200';
files = dir(folder1);

fig1 = figure();
hold on
success = 0;
for num = 3:length(files)
    filename = files(num).name;
    try
        load([folder1, '/', filename, '/all_test_accuracy.mat'])
        plot(all_test_accuracy,'-ms')
        plot(length(all_test_accuracy),all_test_accuracy(end),'-bs','MarkerFaceColor','b')
        success = success + 1;
    catch
        continue
    end
end
grid on
xlabel('Iteration of repairing')
ylabel('Accuracy(%)')
title({'Neural Network Repair',[num2str(success),'/',num2str(num-2),' success'],'Learning Rate 0.001 and Epochs 200'})
hold off


folder2 = './logs_lr0.0002_epochs200';
files = dir(folder2);
fig2 = figure();
hold on
success = 0;
for num = 3:length(files)
    filename = files(num).name;
    try
        load([folder2, '/', filename, '/all_test_accuracy.mat'])
        plot(all_test_accuracy,'-ms')
        plot(length(all_test_accuracy),all_test_accuracy(end),'-bs','MarkerFaceColor','b')
        success = success + 1;
    catch
        continue
    end
end
grid on
xlabel('Iteration of repairing')
ylabel('Accuracy(%)')
title({'Neural Network Repair',[num2str(success),'/',num2str(num-2),' success'],'Learning Rate 0.002 and Epochs 200'})
hold off