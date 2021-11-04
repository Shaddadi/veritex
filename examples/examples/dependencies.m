fprintf('\nINSTALLING NNV....');
fprintf('\nIntalling tbxmanager (requires Matlab R2009a or later) ...');
tbxmanager_folder = 'tbxmanager';

root_folder = pwd();

list = dir;
if ~isfolder(tbxmanager_folder)
    mkdir(tbxmanager_folder);
end

% install mpt toobox and other dependencies
cd(tbxmanager_folder);
urlwrite('https://raw.githubusercontent.com/verivital/tbxmanager/master/tbxmanager.m', 'tbxmanager.m');
%urlwrite('http://www.tbxmanager.com/tbxmanager.m', 'tbxmanager.m');
tbxmanager
savepath
fprintf('\nInstalling tbxmanager toolbox is done!');

cd(root_folder);

fprintf('\nIntalling mpt toolbox and other dependencies...\n');
tbxmanager install mpt mptdoc;
tbxmanager install lcp hysdel cddmex clpmex glpkmex fourier sedumi;
tbxmanager install yalmip; % todo: error due to license, need to force acceptance
fprintf('\nInstalling dependencies is done!');