clc
clear
cd('~/Dropbox/6_Graduate/sim/one-stage-sim-new/')
n = 1000;
ntest = 10000;
sumtab{:} = nan;
c_list = nan(22,1);
obj_list = nan(22,200);
const_list = nan(22,200);
theta1_list = nan(22,200);
theta2_list = nan(22,200);
sim_parm = load('sim_parm');
a1 = [0.5; -0.5; 1];
a2 = [-0.5; 0.5; -1];
for setting = 1:9
  % generate test set for each setting
  % read in simulation setting
  b1 = sim_parm(setting, 1:3)';
  b2 = sim_parm(setting, 4:6)';
  [X, alpha1Hat, beta1Hat, alpha2Hat, beta2Hat, ~, ~] = gen_data(ntest, a1, b1, a2, b2);
  Xint = [ones(ntest,1) X];
  z0 = Xint*alpha1Hat;
  z2_1 = Xint*beta1Hat;
  z2_2 = Xint*beta2Hat;
  for mc = 1:200
      fileName = strcat('output_setting_', num2str(setting), '_mc_', num2str(mc), '.txt');
      result = table2array(readtable(fileName));
      for row = 3:length(result)
        theta1 = result(row,5);
        theta2 = result(row,6);
        theta = [theta1; theta2];
        % z1 = Xint*theta;
        obj_test = analy_objective(theta, X, z0, z2_1, ntest, 1);
        const_test = analy_objective(theta, X, z0, z2_2, ntest, 1);
        c_list(row)  = result(row,1);
        theta1_list(row,mc) = theta1;
        theta2_list(row,mc) = theta2;
        obj_list(row,mc) = obj_test; % result(row,2);
        const_list(row,mc) = const_test; %result(row,3);
     end
  end
  obj_list_mean= mean(obj_list, 2);
  obj_list_std= std(obj_list, 0, 2);
  const_list_mean = mean(const_list, 2);
  const_list_std = std(const_list, 0, 2);
  theta1_list_mean = mean(theta1_list,  2);
  theta1_list_std = std(theta1_list, 0, 2);
  theta2_list_mean = mean(theta2_list, 2);
  theta2_list_std = std(theta2_list, 0, 2);
  set_list = repmat(setting, 22, 1);
  sumtab{setting} = horzcat(set_list, c_list, ...
                            obj_list_mean, obj_list_std, ...
                            const_list_mean, const_list_std, ...
                            theta1_list_mean, theta1_list_std, ...
                            theta2_list_mean, theta2_list_std );
  % txt file
  filename1 = strcat('summary_setting_', num2str(setting), '.txt');
  dlmwrite(filename1, sumtab{setting} , '-append');
  % tex file
  filename2 = strcat('result_', num2str(setting), '.tex');
  FID = fopen(filename2, 'w');
  fprintf(FID, '\\begin{tabular}{rrrrrrrrrr}\\hline \n');
  fprintf(FID, 'setting & $\\nu$  & $\\wh{V}_1(\\wh{\\bs{\\theta}}_{\\nu})$ & $std(\\wh{V}_1)$ & $\\wh{V}_2(\\wh{\\bs{\\theta}}_{\\nu})$ & $std(\\wh{V}_2)$ & $\\wh{\\theta}_{\\nu,1}$ & $std(\\wh{\\theta}_{\\nu,1})$ & $\\wh{\\theta}_{\\nu,2}$ & $std(\\wh{\\theta}_{\\nu,2})$ \\\\ \\hline \n');
  printtab = sumtab{setting};
  for k=5:size(printtab,1)
      printline = printtab(k, :);
      fprintf(FID, '%d & %8.2f & %8.2f & %8.2f  & %8.2f &  %8.2f &  %8.2f &  %8.2f &  %8.2f &  %8.2f \\\\ ', printline);
      if k==size(printtab,1)
          fprintf(FID, '\\hline ');
      end
      fprintf(FID, '\n');
  end
  fprintf(FID, '\\end{tabular}\n');
  fclose(FID);

  % plot
  width=10;
  height=8;
  x0=1;
  y0=1;
  figure('Units','inches', 'Position',[x0 y0 width height],...
  'PaperPositionMode','auto');
% (x0,y0) = position of the lower left side of the figure
   c = printtab(:,2);
   y1 = printtab(:,3);
   std1 = printtab(:,4);
   y1U = y1 + 1.96*std1/sqrt(n);
   y1L = y1 - 1.96*std1/sqrt(n);
   y2 = printtab(:,5);
   std2 = printtab(:,6);
   y2U = y2 + 1.96*std2/sqrt(n);
   y2L = y2 - 1.96*std2/sqrt(n);
   plot(c,y1,'--ro',c,y2,'-.bo','LineWidth',1.5);
%    hold on;
%    plot(c,y1U,':', c,y1L,':', c,y2U,':',c,y2L,':','LineWidth',1.5);
%    hold off;
   hline = refline(1,0);
   set(hline,'LineStyle',':', 'LineWidth',1.5);
   xlabel({'$\nu$ bound on secondary potential outcome'}, ...
             'interpreter' ,'latex', 'FontSize',15 )
   ylabel({'$\widehat{V}$ values of estimated constrained optimal regimes'},...
            'interpreter' ,'latex', 'FontSize',15 )
   title({'Efficient Frontier Plot $\widehat{V}_1$ / $\widehat{V}_2$ vs. $\nu$'},...
          'interpreter' ,'latex', 'FontSize',15);
   legend({'$\widehat{V}_1$ vs. $\nu$',  '$\widehat{V}_2$ vs. $\nu$'}, ...
             'interpreter' ,'latex', 'Location','SouthEast','FontSize',15);
  % axis([0 t(end) -1.5 1.5]);
   set(gca, 'Units','normalized', ...
       'FontUnits','points',...
       'FontWeight','normal',...
       'FontSize',15);
   print(strcat('efficient_plot', num2str(setting)), '-dpdf', '-bestfit' ) ;
end
