TFC_load_data;

%%%%%%%%%%%%%%%%%%%%%%%%%%
% Binned freezing plot TFC
%%%%%%%%%%%%%%%%%%%%%%%%%%

f=figure; hold on;
if for_paper
	f.PaperPosition = [0 0 1.25 0.75];
else
	f.PaperPosition = [0 0 6 4];
end
file_str = '1a_TFC_binned_freezing';

FSTI_TFC_mean = mean(FSTI_TFC_tot,1)
FSTE_TFC_mean = mean(FSTE_TFC_tot,1)
FSTC_TFC_mean = mean(FSTC_TFC_tot,1)

FSTI_TFC_sem = std(FSTI_TFC_tot,1)./sqrt(size(FSTI_TFC_tot,1));
FSTE_TFC_sem = std(FSTE_TFC_tot,1)./sqrt(size(FSTE_TFC_tot,1));
FSTC_TFC_sem = std(FSTC_TFC_tot,1)./sqrt(size(FSTC_TFC_tot,1));

%plot(onset,FSTI_mean,'bo-');
%plot(onset,FSTE_mean,'ro-');
%plot(onset,FSTC_mean,'ko-');
e = errorbar(onset_tfc, FSTI_TFC_mean, FSTI_TFC_sem, 'o-', "Color", my_b, "MarkerFaceColor", my_b, "LineWidth", my_linewidth, "MarkerSize", my_markersize); e.CapSize=0;
e = errorbar(onset_tfc, FSTE_TFC_mean, FSTE_TFC_sem, 'o-', "Color", my_r, "MarkerFaceColor", my_r, "LineWidth", my_linewidth, "MarkerSize", my_markersize); e.CapSize=0;
e = errorbar(onset_tfc, FSTC_TFC_mean, FSTC_TFC_sem, 'o-', "Color", my_k, "MarkerFaceColor", my_k, "LineWidth", my_linewidth, "MarkerSize", my_markersize); e.CapSize=0;

for i=1:length(sound_onsets)
	a= fill([sound_onsets(i) sound_onsets(i) sound_offsets(i) sound_offsets(i)], [0 85 85 0], my_light_grey);
	a.FaceAlpha = 0.1;
	a.LineStyle = "none";
end

for i=1:length(shock_onsets)
	a= fill([shock_onsets(i) shock_onsets(i) shock_offsets(i) shock_offsets(i)], [0 85 85 0], my_r);
	a.LineStyle = "none";
end

%{
xline(180)
xline(420)
xline(660)
xline(200)
xline(440)
xline(680)
%}

if ~for_paper
	ylabel('Freezing (%)');
end
xticks([60:120:floor(max(onset_tfc))]);
xticklabels({1:2:floor(max(onset_tfc/60))});
ax = gca;
%ax.LineWidth=1.5;
ax.LineWidth=my_linewidth;
ax.TickDir = 'out';
set(gca,'TickDir','out'); 
%ax=gca; ax.XAxis.MajorTickChild.LineWidth = 1
%set(gca,'LineWidth',2,'TickLength',[0.025 0.025]);

if ~for_paper
	xlabel('Time (min)');
end
xlim([0 max(onset_tfc)]);
ylim([0 85]);

%legend('FSTI_1wk','FSTE_1wk','FSTC_1wk');
%legend('SST inhibition','SST activation','SST control');
%%%%title('Test B (1 week)');
if ~for_paper
	title('Conditioning');
end

%{

TODO

[time_bins, groups, freeze] = anovan_TFC(FSTC_TFC_tot, FSTE_TFC_tot, FSTI_TFC_tot, post_tone_only);
[p,tbl,stats] = anovan(freeze,{groups time_bins},'model',2,'varnames',{'groups','time bins'});
[c,m,h,gnames] = multcompare(stats,'alpha',.05,'CType','bonferroni')
%}

if saveit
	print(f, sprintf('%s\\%s.svg',file_path,file_str), '-dsvg');
	print(f, sprintf('%s\\%s.png',file_path,file_str), '-dpng', '-r300');		
	disp(sprintf("done."));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Binned freezing plot TFC-HC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%FSTE_HC_tot = [FSTH1];
%FSTE_HC_1wk_tot = [FSTH1_1wk];
%FSTE_HC_TestA_tot = [FSTH1_TestA];
%FSTE_HC_TestA_1wk_tot = [FSTH1_TestA_1wk];
FSTE_HC_TFC_tot = [FSTH1_TFC];

f=figure; hold on;
if for_paper
	f.PaperPosition = [0 0 1.25 0.75];
else
	f.PaperPosition = [0 0 6 4];
end
file_str = '1a_TFC_binned_freezing_HC';

FSTI_TFC_mean = mean(FSTI_TFC_tot,1)
FSTE_TFC_mean = mean(FSTE_TFC_tot,1)
FSTC_TFC_mean = mean(FSTC_TFC_tot,1)

FSTI_TFC_sem = std(FSTI_TFC_tot,1)./sqrt(size(FSTI_TFC_tot,1));
FSTE_TFC_sem = std(FSTE_TFC_tot,1)./sqrt(size(FSTE_TFC_tot,1));
FSTC_TFC_sem = std(FSTC_TFC_tot,1)./sqrt(size(FSTC_TFC_tot,1));

%plot(onset,FSTI_mean,'bo-');
%plot(onset,FSTE_mean,'ro-');
%plot(onset,FSTC_mean,'ko-');
e = errorbar(onset_tfc, FSTI_TFC_mean, FSTI_TFC_sem, 'o-', "Color", my_b, "MarkerFaceColor", my_b, "LineWidth", my_linewidth, "MarkerSize", my_markersize); e.CapSize=0;
e = errorbar(onset_tfc, FSTE_TFC_mean, FSTE_TFC_sem, 'o-', "Color", my_r, "MarkerFaceColor", my_r, "LineWidth", my_linewidth, "MarkerSize", my_markersize); e.CapSize=0;
e = errorbar(onset_tfc, FSTC_TFC_mean, FSTC_TFC_sem, 'o-', "Color", my_k, "MarkerFaceColor", my_k, "LineWidth", my_linewidth, "MarkerSize", my_markersize); e.CapSize=0;

for i=1:length(sound_onsets)
	a= fill([sound_onsets(i) sound_onsets(i) sound_offsets(i) sound_offsets(i)], [0 85 85 0], my_light_grey);
	a.FaceAlpha = 0.1;
	a.LineStyle = "none";
end

for i=1:length(shock_onsets)
	a= fill([shock_onsets(i) shock_onsets(i) shock_offsets(i) shock_offsets(i)], [0 85 85 0], my_r);
	a.LineStyle = "none";
end

%{
xline(180)
xline(420)
xline(660)
xline(200)
xline(440)
xline(680)
%}

if ~for_paper
	ylabel('Freezing (%)');
end
xticks([60:120:floor(max(onset_tfc))]);
xticklabels({1:2:floor(max(onset_tfc/60))});
ax = gca;
%ax.LineWidth=1.5;
ax.LineWidth=my_linewidth;
ax.TickDir = 'out';
set(gca,'TickDir','out'); 
%ax=gca; ax.XAxis.MajorTickChild.LineWidth = 1
%set(gca,'LineWidth',2,'TickLength',[0.025 0.025]);

if ~for_paper
	xlabel('Time (min)');
end
xlim([0 max(onset_tfc)]);
ylim([0 85]);

%legend('FSTI_1wk','FSTE_1wk','FSTC_1wk');
%legend('SST inhibition','SST activation','SST control');
%%%%title('Test B (1 week)');
if ~for_paper
	title('Conditioning');
end

%{

TODO

[time_bins, groups, freeze] = anovan_TFC(FSTC_TFC_tot, FSTE_TFC_tot, FSTI_TFC_tot, post_tone_only);
[p,tbl,stats] = anovan(freeze,{groups time_bins},'model',2,'varnames',{'groups','time bins'});
[c,m,h,gnames] = multcompare(stats,'alpha',.05,'CType','bonferroni')
%}

if saveit
	print(f, sprintf('%s\\%s.svg',file_path,file_str), '-dsvg');
	print(f, sprintf('%s\\%s.png',file_path,file_str), '-dpng', '-r300');		
	disp(sprintf("done."));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Binned freezing plot 48 hours
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f=figure; hold on;
if for_paper
	f.PaperPosition = [0 0 3 2];
else
	f.PaperPosition = [0 0 6 4];
end

file_str = '2a_48hr_binned_freezing';

FSTI_mean = mean(FSTI_tot,1)
FSTE_mean = mean(FSTE_tot,1)
FSTC_mean = mean(FSTC_tot,1)

FSTI_sem = std(FSTI_tot,1)./sqrt(size(FSTI_tot,1));
FSTE_sem = std(FSTE_tot,1)./sqrt(size(FSTE_tot,1));
FSTC_sem = std(FSTC_tot,1)./sqrt(size(FSTC_tot,1));

%plot(onset,FSTI_mean,'bo-');
%plot(onset,FSTE_mean,'ro-');
%plot(onset,FSTC_mean,'ko-');
e = errorbar(onset, FSTI_mean, FSTI_sem, 'o-', "Color", my_b, "MarkerFaceColor", my_b, "LineWidth", my_linewidth); e.CapSize=0;
e = errorbar(onset, FSTE_mean, FSTE_sem, 'o-', "Color", my_r, "MarkerFaceColor", my_r, "LineWidth", my_linewidth); e.CapSize=0;
e = errorbar(onset, FSTC_mean, FSTC_sem, 'o-', "Color", my_k, "MarkerFaceColor", my_k, "LineWidth", my_linewidth); e.CapSize=0;

for i=1:length(sound_onsets_test)
	a= fill([sound_onsets(i) sound_onsets(i) sound_offsets(i) sound_offsets(i)], [0 70 70 0], my_light_grey);
	a.FaceAlpha = 0.1;
	a.LineStyle = "none";
end

%{
xline(180)
xline(420)
xline(660)
xline(200)
xline(440)
xline(680)
%}

ylabel('Freezing (%)');
xticks([60:120:floor(max(onset))]);
xticklabels({1:2:floor(max(onset/60))});
ax = gca;
%ax.LineWidth=1.5;
ax.LineWidth=my_linewidth;
ax.TickDir = 'out';
set(gca,'TickDir','out'); 
%ax=gca; ax.XAxis.MajorTickChild.LineWidth = 1
%set(gca,'LineWidth',2,'TickLength',[0.025 0.025]);

xlabel('Time (min)');
xlim([0 max(onset)]);
ylim([0 70]);

%legend('FSTI_1wk','FSTE_1wk','FSTC_1wk');
%legend('SST inhibition','SST activation','SST control');
%%%%title('Test B (1 week)');
title('Test 48hr');

%{

TODO

[time_bins, groups, freeze] = anovan_TFC(FSTC_TFC_tot, FSTE_TFC_tot, FSTI_TFC_tot, tone_post_tone);
[~,~,stats] = anovan(freeze,{groups time_bins},'model',2,'varnames',{'groups','time bins'});
[c,m,h,gnames] = multcompare(stats)
%}

if saveit
	print(f, sprintf('%s\\%s.svg',file_path,file_str), '-dsvg');
	print(f, sprintf('%s\\%s.png',file_path,file_str), '-dpng');		
	disp(sprintf("done."));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Binned freezing plot 1wk
%%%%%%%%%%%%%%%%%%%%%%%%%%%

f=figure; hold on;
if for_paper
	f.PaperPosition = [0 0 3 2];
else
	f.PaperPosition = [0 0 6 4];
end
%set(gca,'FontSize',18);
file_str = '3a_1wk_binned_freezing';

FSTI_1wk_mean = mean(FSTI_1wk_tot,1)
FSTE_1wk_mean = mean(FSTE_1wk_tot,1)
FSTC_1wk_mean = mean(FSTC_1wk_tot,1)

FSTI_1wk_sem = std(FSTI_1wk_tot,1)./sqrt(size(FSTI_1wk_tot,1));
FSTE_1wk_sem = std(FSTE_1wk_tot,1)./sqrt(size(FSTE_1wk_tot,1));
FSTC_1wk_sem = std(FSTC_1wk_tot,1)./sqrt(size(FSTC_1wk_tot,1));

%plot(onset,FSTI_mean,'bo-');
%plot(onset,FSTE_mean,'ro-');
%plot(onset,FSTC_mean,'ko-');
e = errorbar(onset_1wk, FSTI_1wk_mean, FSTI_1wk_sem, 'o-', "Color", my_b, "MarkerFaceColor", my_b, "LineWidth", my_linewidth); e.CapSize=0;
e = errorbar(onset_1wk, FSTE_1wk_mean, FSTE_1wk_sem, 'o-', "Color", my_r, "MarkerFaceColor", my_r, "LineWidth", my_linewidth); e.CapSize=0;
e = errorbar(onset_1wk, FSTC_1wk_mean, FSTC_1wk_sem, 'o-', "Color", my_k, "MarkerFaceColor", my_k, "LineWidth", my_linewidth); e.CapSize=0;

%plot(onset_1wk,FSTI_1wk_mean,'bo-');
%plot(onset_1wk,FSTE_1wk_mean,'ro-');
%plot(onset_1wk,FSTC_1wk_mean,'ko-');


for i=1:length(sound_onsets_test)
	a= fill([sound_onsets(i) sound_onsets(i) sound_offsets(i) sound_offsets(i)], [0 70 70 0], my_light_grey);
	a.FaceAlpha = 0.1;
	a.LineStyle = "none";
end	
%{
xline(180)
xline(420)
xline(660)
xline(200)
xline(440)
xline(680)
%}
ylabel('Freezing (%)');
xticks([60:120:floor(max(onset_1wk))]);
xticklabels({1:2:floor(max(onset_1wk/60))});
ax = gca;
%ax.LineWidth=1.5;
ax.LineWidth=my_linewidth;
ax.TickDir = 'out';
set(gca,'TickDir','out'); 
%ax=gca; ax.XAxis.MajorTickChild.LineWidth = 1
%set(gca,'LineWidth',2,'TickLength',[0.025 0.025]);

xlabel('Time (min)');
xlim([0 max(onset_1wk)]);
ylim([0 70]);

%legend('FSTI_1wk','FSTE_1wk','FSTC_1wk');
%legend('SST inhibition','SST activation','SST control');
%%%%title('Test B (1 week)');
title('Test 1wk');

if saveit
	print(f, sprintf('%s\\%s.svg',file_path,file_str), '-dsvg');
	print(f, sprintf('%s\\%s.png',file_path,file_str), '-dpng');		
	disp(sprintf("done."));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate period-related average freeze scores
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% For TFC
%
%FSTI_TFC_mean

tone_FSTI = mean(FSTI_tot(:,tone),2);
tone_FSTE = mean(FSTE_tot(:,tone),2);
tone_FSTC = mean(FSTC_tot(:,tone),2);
mean_tone_FSTI = mean(mean(FSTI_tot(:,tone),2));
mean_tone_FSTE = mean(mean(FSTE_tot(:,tone),2));
mean_tone_FSTC = mean(mean(FSTC_tot(:,tone),2));
std_tone_FSTI = std(mean(FSTI_tot(:,tone),2));
std_tone_FSTE = std(mean(FSTE_tot(:,tone),2));
std_tone_FSTC = std(mean(FSTC_tot(:,tone),2));
sem_tone_FSTI = std_tone_FSTI./sqrt(size(FSTI_tot,1));
sem_tone_FSTE = std_tone_FSTE./sqrt(size(FSTE_tot,1));
sem_tone_FSTC = std_tone_FSTC./sqrt(size(FSTC_tot,1));

tone_post_tone_FSTI = mean(FSTI_tot(:,tone_post_tone),2);
tone_post_tone_FSTE = mean(FSTE_tot(:,tone_post_tone),2);
tone_post_tone_FSTC = mean(FSTC_tot(:,tone_post_tone),2);
mean_tone_post_tone_FSTI = mean(mean(FSTI_tot(:,tone_post_tone),2));
mean_tone_post_tone_FSTE = mean(mean(FSTE_tot(:,tone_post_tone),2));
mean_tone_post_tone_FSTC = mean(mean(FSTC_tot(:,tone_post_tone),2));
std_tone_post_tone_FSTI = std(mean(FSTI_tot(:,tone_post_tone),2));
std_tone_post_tone_FSTE = std(mean(FSTE_tot(:,tone_post_tone),2));
std_tone_post_tone_FSTC = std(mean(FSTC_tot(:,tone_post_tone),2));
sem_tone_post_tone_FSTI = std_tone_post_tone_FSTI./sqrt(size(FSTI_tot,1));
sem_tone_post_tone_FSTE = std_tone_post_tone_FSTE./sqrt(size(FSTE_tot,1));
sem_tone_post_tone_FSTC = std_tone_post_tone_FSTC./sqrt(size(FSTC_tot,1));

post_tone_only_FSTI = mean(FSTI_tot(:,post_tone_only),2);
post_tone_only_FSTE = mean(FSTE_tot(:,post_tone_only),2);
post_tone_only_FSTC = mean(FSTC_tot(:,post_tone_only),2);
mean_post_tone_only_FSTI = mean(mean(FSTI_tot(:,post_tone_only),2));
mean_post_tone_only_FSTE = mean(mean(FSTE_tot(:,post_tone_only),2));
mean_post_tone_only_FSTC = mean(mean(FSTC_tot(:,post_tone_only),2));
std_post_tone_only_FSTI = std(mean(FSTI_tot(:,post_tone_only),2));
std_post_tone_only_FSTE = std(mean(FSTE_tot(:,post_tone_only),2));
std_post_tone_only_FSTC = std(mean(FSTC_tot(:,post_tone_only),2));
sem_post_tone_only_FSTI = std_post_tone_only_FSTI./sqrt(size(FSTI_tot,1));
sem_post_tone_only_FSTE = std_post_tone_only_FSTE./sqrt(size(FSTE_tot,1));
sem_post_tone_only_FSTC = std_post_tone_only_FSTC./sqrt(size(FSTC_tot,1));

first_3min_FSTI = mean(FSTI_tot(:,first_3min),2);
first_3min_FSTE = mean(FSTE_tot(:,first_3min),2);
first_3min_FSTC = mean(FSTC_tot(:,first_3min),2);
mean_first_3min_FSTI = mean(mean(FSTI_tot(:,first_3min),2));
mean_first_3min_FSTE = mean(mean(FSTE_tot(:,first_3min),2));
mean_first_3min_FSTC = mean(mean(FSTC_tot(:,first_3min),2));
std_first_3min_FSTI = std(mean(FSTI_tot(:,first_3min),2));
std_first_3min_FSTE = std(mean(FSTE_tot(:,first_3min),2));
std_first_3min_FSTC = std(mean(FSTC_tot(:,first_3min),2));
sem_first_3min_FSTI = std_first_3min_FSTI./sqrt(size(FSTI_tot,1));
sem_first_3min_FSTE = std_first_3min_FSTE./sqrt(size(FSTE_tot,1));
sem_first_3min_FSTC = std_first_3min_FSTC./sqrt(size(FSTC_tot,1));

% maybe delete?
mean_post_tone_20s_FSTI = mean(mean(FSTI_tot(:,post_tone_20s),2));
mean_post_tone_20s_FSTE = mean(mean(FSTE_tot(:,post_tone_20s),2));
mean_post_tone_20s_FSTC = mean(mean(FSTC_tot(:,post_tone_20s),2));
std_post_tone_20s_FSTI = std(mean(FSTI_tot(:,post_tone_20s),2));
std_post_tone_20s_FSTE = std(mean(FSTE_tot(:,post_tone_20s),2));
std_post_tone_20s_FSTC = std(mean(FSTC_tot(:,post_tone_20s),2));
sem_post_tone_20s_FSTI = std_post_tone_20s_FSTI./sqrt(size(FSTI_tot,1));
sem_post_tone_20s_FSTE = std_post_tone_20s_FSTE./sqrt(size(FSTE_tot,1));
sem_post_tone_20s_FSTC = std_post_tone_20s_FSTC./sqrt(size(FSTC_tot,1));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 48 hours period-averaged freezing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The plot - general

%%% OBSOLETE

figure; hold on;
bar(1,mean_tone_FSTE,'r');
bar(2,mean_tone_FSTI,'b');
bar(3,mean_tone_FSTC,'k');
bar(5,mean_tone_post_tone_FSTE,'r');
bar(6,mean_tone_post_tone_FSTI,'b');
bar(7,mean_tone_post_tone_FSTC,'k');
bar(9,mean_post_tone_only_FSTE,'r');
bar(10,mean_post_tone_only_FSTI,'b');
bar(11,mean_post_tone_only_FSTC,'k');
bar(13,mean_post_tone_20s_FSTE,'r');
bar(14,mean_post_tone_20s_FSTI,'b');
bar(15,mean_post_tone_20s_FSTC,'k');

set(gca,'XTick',[1:15]);
set(gca,'XTickLabels',{'Tone','Tone','Tone','','Tone+post-tone','Tone+post-tone','Tone+post-tone','','Post-tone only','Post-tone only','Post-tone only','','Post-tone 20s','Post-tone 20s','Post-tone 20s'});
set(gca,'XTickLabelRotation',-45);
legend({'SST Exc','SST Inh','SST Ctl'});
title('Test B (48 hours)');
ylabel('Freezing (%)');

%%%% /OBSOLETE

% The plot - paper

%
% Test B 
%
f=figure; hold on;
%use_test = 'anova'
%use_test = 'ranksum'
%use_test = 'kw';
%use_test = 'ttest2'

%post_hoc = 'hsd'
%post_hoc = 'bonferroni'
%post_hoc = 'dunn-sidak'

if for_paper
	f.PaperPosition = [0 0 2 2];
else
	f.PaperPosition = [0 0 4 4];
end
%set(gca,'FontSize',18);
file_str = '4a_48hr_period_freezing';

data_E = mean(FSTE_tot(:,tone),2);
data_I = mean(FSTI_tot(:,tone),2);
data_C = mean(FSTC_tot(:,tone),2);
[G, P] = do_stats_tfc(data_E, data_I, data_C, use_test, post_hoc, 0);

%sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',14,'nosort',1,'max_errbar',max_sem));

b = bar([1 2 3], [mean_tone_FSTE; mean_tone_FSTI; mean_tone_FSTC], 1.0); b.FaceColor='flat'; b.CData(1,:)=my_r; b.CData(2,:)=my_b, b.CData(3,:)=my_k; 
max_sem = max([sem_tone_FSTE, sem_tone_FSTI, sem_tone_FSTC]);
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',8,'nosort',1,'max_errbar',max_sem));
%e = errorbar([1 2 3], [mean_tone_FSTE; mean_tone_FSTI; mean_tone_FSTC], [std_tone_FSTE; std_tone_FSTI; std_tone_FSTC], 'k.'); e.CapSize=0; e.LineWidth=2;
e = errorbar(1, mean_tone_FSTE, sem_tone_FSTE, 'Color',my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(2, mean_tone_FSTI, sem_tone_FSTI, 'Color',my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(3, mean_tone_FSTC, sem_tone_FSTC, 'Color',my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;

data_E = mean(FSTE_tot(:,post_tone_only),2);
data_I = mean(FSTI_tot(:,post_tone_only),2);
data_C = mean(FSTC_tot(:,post_tone_only),2);
[G, P] = do_stats_tfc(data_E, data_I, data_C, use_test, post_hoc, 4);

b = bar([5 6 7], [mean_post_tone_only_FSTE; mean_post_tone_only_FSTI; mean_post_tone_only_FSTC], 1.0); b.FaceColor='flat'; b.CData(1,:)=my_r; b.CData(2,:)=my_b, b.CData(3,:)=my_k; 
max_sem = max([sem_post_tone_only_FSTE, sem_post_tone_only_FSTI, sem_post_tone_only_FSTC]);
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',8,'nosort',1,'max_errbar',max_sem));
%e = errorbar([1 2 3], [mean_tone_FSTE; mean_tone_FSTI; mean_tone_FSTC], [std_tone_FSTE; std_tone_FSTI; std_tone_FSTC], 'k.'); e.CapSize=0; e.LineWidth=2;
e = errorbar(5, mean_post_tone_only_FSTE, sem_post_tone_only_FSTE, 'Color',my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(6, mean_post_tone_only_FSTI, sem_post_tone_only_FSTI, 'Color',my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(7, mean_post_tone_only_FSTC, sem_post_tone_only_FSTC, 'Color',my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;

%{
names = {'SST inhibition', 'SST activation', 'SST control   '};
data = {post_tone_only_FSTI, post_tone_only_FSTE, post_tone_only_FSTC};
stats_anova1(data, names, 'ANOVA');
stats_kruskalwallis(data, names, 'KW');
%}

set(gca,'XTick',[1:7]);
%set(gca,'XTickLabels',{'Tone','Tone','Tone', '', 'Post tone','Post tone','Post tone'});
%set(gca,'XTickLabels',{'Exc-Tone','Inh-Tone','Ctl-Tone', '', 'Exc-ITI ','Inh-ITI ','Ctl-ITI '});
set(gca,'XTickLabels',{'Exc          ','Inh     ','Ctl     ', '', 'Exc     ','Inh     ','Ctl     '});
set(gca,'XTickLabelRotation',-45);

ylabel('Freezing (%)');
title('Test 48hr');

if saveit
	print(f, sprintf('%s\\%s.svg',file_path,file_str), '-dsvg');
	print(f, sprintf('%s\\%s.png',file_path,file_str), '-dpng');		
	disp(sprintf("done."));
end

% The plot - paper - 48hr (same as above) but with first_3min

f=figure; hold on;
%use_test = 'anova'
%use_test = 'ranksum'
%use_test = 'kw';

%post_hoc = 'hsd'
%post_hoc = 'bonferroni'
%post_hoc = 'dunn-sidak'

if for_paper
	f.PaperPosition = [0 0 3 2];
else
	f.PaperPosition = [0 0 6 4];
end
%set(gca,'FontSize',18);
file_str = '4b_48hr_period_freezing_first_3min';

data_E = mean(FSTE_tot(:,tone),2);
data_I = mean(FSTI_tot(:,tone),2);
data_C = mean(FSTC_tot(:,tone),2);
[G, P] = do_stats_tfc(data_E, data_I, data_C, use_test, post_hoc, 0);

b = bar([1 2 3], [mean_tone_FSTE; mean_tone_FSTI; mean_tone_FSTC], 1.0); b.FaceColor='flat'; b.CData(1,:)=my_r; b.CData(2,:)=my_b, b.CData(3,:)=my_k; 
max_sem = max([sem_tone_FSTE, sem_tone_FSTI, sem_tone_FSTC]);
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',8,'nosort',1,'max_errbar',max_sem));
%e = errorbar([1 2 3], [mean_tone_FSTE; mean_tone_FSTI; mean_tone_FSTC], [std_tone_FSTE; std_tone_FSTI; std_tone_FSTC], 'k.'); e.CapSize=0; e.LineWidth=2;
e = errorbar(1, mean_tone_FSTE, sem_tone_FSTE, 'Color',my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(2, mean_tone_FSTI, sem_tone_FSTI, 'Color',my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(3, mean_tone_FSTC, sem_tone_FSTC, 'Color',my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;

data_E = mean(FSTE_tot(:,post_tone_only),2);
data_I = mean(FSTI_tot(:,post_tone_only),2);
data_C = mean(FSTC_tot(:,post_tone_only),2);
[G, P] = do_stats_tfc(data_E, data_I, data_C, use_test, post_hoc, 4);

b = bar([5 6 7], [mean_post_tone_only_FSTE; mean_post_tone_only_FSTI; mean_post_tone_only_FSTC], 1.0); b.FaceColor='flat'; b.CData(1,:)=my_r; b.CData(2,:)=my_b, b.CData(3,:)=my_k; 
max_sem = max([sem_post_tone_only_FSTE, sem_post_tone_only_FSTI, sem_post_tone_only_FSTC]);
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',8,'nosort',1,'max_errbar',max_sem));
%e = errorbar([1 2 3], [mean_tone_FSTE; mean_tone_FSTI; mean_tone_FSTC], [std_tone_FSTE; std_tone_FSTI; std_tone_FSTC], 'k.'); e.CapSize=0; e.LineWidth=2;
e = errorbar(5, mean_post_tone_only_FSTE, sem_post_tone_only_FSTE, 'Color',my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(6, mean_post_tone_only_FSTI, sem_post_tone_only_FSTI, 'Color',my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(7, mean_post_tone_only_FSTC, sem_post_tone_only_FSTC, 'Color',my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
[p_EI, h_EI] = ranksum(mean(FSTE_tot(:,post_tone_only),2), mean(FSTI_tot(:,post_tone_only),2));
[p_EC, h_EC] = ranksum(mean(FSTE_tot(:,post_tone_only),2), mean(FSTC_tot(:,post_tone_only),2));
[p_IC, h_IC] = ranksum(mean(FSTI_tot(:,post_tone_only),2), mean(FSTC_tot(:,post_tone_only),2));

data_E = mean(FSTE_tot(:,post_tone_only),2);
data_I = mean(FSTI_tot(:,post_tone_only),2);
data_C = mean(FSTC_tot(:,post_tone_only),2);
[G, P] = do_stats_tfc(data_E, data_I, data_C, use_test, post_hoc, 8);

b = bar([9 10 11], [mean_first_3min_FSTE; mean_first_3min_FSTI; mean_first_3min_FSTC], 1.0); b.FaceColor='flat'; b.CData(1,:)=my_r; b.CData(2,:)=my_b, b.CData(3,:)=my_k; 
%e = errorbar([1 2 3], [mean_tone_FSTE; mean_tone_FSTI; mean_tone_FSTC], [std_tone_FSTE; std_tone_FSTI; std_tone_FSTC], 'k.'); e.CapSize=0; e.LineWidth=2;
max_sem = max([sem_first_3min_FSTE, sem_first_3min_FSTI, sem_first_3min_FSTC]);
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',8,'nosort',1,'max_errbar',max_sem));
e = errorbar(9, mean_first_3min_FSTE, sem_first_3min_FSTE, 'Color',my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(10, mean_first_3min_FSTI, sem_first_3min_FSTI, 'Color',my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(11, mean_first_3min_FSTC, sem_first_3min_FSTC, 'Color',my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
[p_EI, h_EI] = ranksum(mean(FSTE_tot(:,post_tone_only),2), mean(FSTI_tot(:,post_tone_only),2));
[p_EC, h_EC] = ranksum(mean(FSTE_tot(:,post_tone_only),2), mean(FSTC_tot(:,post_tone_only),2));
[p_IC, h_IC] = ranksum(mean(FSTI_tot(:,post_tone_only),2), mean(FSTC_tot(:,post_tone_only),2));

set(gca,'XTick',[1:11]);
set(gca,'XTickLabels',{'Tone','Tone','Tone', '', 'Post tone','Post tone','Post tone', '',strcat('First ', first_min), strcat('First ', first_min), strcat('First ', first_min)});
set(gca,'XTickLabelRotation',-45);

title('Test 48hr');

if saveit
	print(f, sprintf('%s\\%s.svg',file_path,file_str), '-dsvg');
	print(f, sprintf('%s\\%s.png',file_path,file_str), '-dpng');		
	disp(sprintf("done."));
end

% Compare across periods (Test B 48hr)

f=figure; hold on;

if for_paper
	f.PaperPosition = [0 0 3 2];
else
	f.PaperPosition = [0 0 6 4];
end
%set(gca,'FontSize',18);
file_str = '4c_48hr_period_freezing_first_3min_comparison';

b = bar([1 2 3], [mean_first_3min_FSTE; mean_tone_FSTE; mean_post_tone_only_FSTE], 1.0); b.FaceColor='flat'; b.CData(1,:)=my_r; b.CData(2,:)=my_r, b.CData(3,:)=my_r;
e = errorbar(1, mean_first_3min_FSTE, sem_first_3min_FSTE, 'Color',my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(2, mean_tone_FSTE, sem_tone_FSTE, 'Color',my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(3, mean_post_tone_only_FSTE, sem_post_tone_only_FSTE, 'Color',my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
scatter(ones(1,length(first_3min_FSTE)), first_3min_FSTE, my_sz,my_k);
scatter(ones(1,length(tone_FSTE))+1, tone_FSTE, my_sz,my_k);
scatter(ones(1,length(post_tone_only_FSTE))+2, post_tone_only_FSTE, my_sz,my_k);
for i=1:length(first_3min_FSTE)
	plot([1 2], [first_3min_FSTE(i), tone_FSTE(i)], 'k-');
	plot([2 3], [tone_FSTE(i), post_tone_only_FSTE(i)], 'k-');
end

max_sem = max([sem_first_3min_FSTE, sem_tone_FSTE, sem_post_tone_only_FSTE]);
data = {mean(FSTE_tot(:,first_3min),2), mean(FSTE_tot(:,tone),2), mean(FSTE_tot(:,post_tone_only),2)};
[G, P] = stats_ranova(data, post_hoc, 0);
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',14,'nosort',1,'max_errbar',max_sem));

b = bar([5 6 7], [mean_first_3min_FSTI; mean_tone_FSTI; mean_post_tone_only_FSTI], 1.0); b.FaceColor='flat'; b.CData(1,:)=my_b; b.CData(2,:)=my_b, b.CData(3,:)=my_b;
e = errorbar(5, mean_first_3min_FSTI, sem_first_3min_FSTI, 'Color',my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(6, mean_tone_FSTI, sem_tone_FSTI, 'Color',my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(7, mean_post_tone_only_FSTI, sem_post_tone_only_FSTI, 'Color',my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
scatter(ones(1,length(first_3min_FSTI))+4, first_3min_FSTI, my_sz,my_k);
scatter(ones(1,length(tone_FSTI))+5, tone_FSTI, my_sz,my_k);
scatter(ones(1,length(post_tone_only_FSTI))+6, post_tone_only_FSTI, my_sz,my_k);
for i=1:length(first_3min_FSTI)
	plot([5 6], [first_3min_FSTI(i), tone_FSTI(i)], 'k-');
	plot([6 7], [tone_FSTI(i), post_tone_only_FSTI(i)], 'k-');
end

max_sem = max([sem_first_3min_FSTE, sem_tone_FSTE, sem_post_tone_only_FSTE]);
data = {mean(FSTI_tot(:,first_3min),2), mean(FSTI_tot(:,tone),2), mean(FSTI_tot(:,post_tone_only),2)};
[G, P] = stats_ranova(data, post_hoc, 4);
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',14,'nosort',1,'max_errbar',max_sem));

b = bar([9 10 11], [mean_first_3min_FSTC; mean_tone_FSTC; mean_post_tone_only_FSTC], 1.0); b.FaceColor='flat'; b.CData(1,:)=my_k; b.CData(2,:)=my_k, b.CData(3,:)=my_k;
e = errorbar(9, mean_first_3min_FSTC, sem_first_3min_FSTC, 'Color',my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(10, mean_tone_FSTC, sem_tone_FSTC, 'Color',my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(11, mean_post_tone_only_FSTC, sem_post_tone_only_FSTC, 'Color',my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
scatter(ones(1,length(first_3min_FSTC))+8, first_3min_FSTC, my_sz,my_h);
scatter(ones(1,length(tone_FSTC))+9, tone_FSTC, my_sz,my_h);
scatter(ones(1,length(post_tone_only_FSTC))+10, post_tone_only_FSTC, my_sz,my_h);
for i=1:length(first_3min_FSTC)
	plot([9 10], [first_3min_FSTC(i), tone_FSTC(i)], 'Color',my_h);
	plot([10 11], [tone_FSTC(i), post_tone_only_FSTC(i)], 'Color',my_h);
end

max_sem = max([sem_first_3min_FSTC, sem_tone_FSTC, sem_post_tone_only_FSTC]);
data = {mean(FSTC_tot(:,first_3min),2), mean(FSTC_tot(:,tone),2), mean(FSTC_tot(:,post_tone_only),2)};
[G, P] = stats_ranova(data, post_hoc, 8);
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',14,'nosort',1,'max_errbar',max_sem));

set(gca,'XTick',[1:11]);
set(gca,'XTickLabels',{strcat('First ', first_min),'Tone','Post tone', '', strcat('First ', first_min),'Tone','Post tone', '', strcat('First ', first_min),'Tone','Post tone'});
set(gca,'XTickLabelRotation',-45);
ylabel('Freezing (%)');
title('Test 48hr - within groups');

if saveit
	print(f, sprintf('%s\\%s.svg',file_path,file_str), '-dsvg');
	print(f, sprintf('%s\\%s.png',file_path,file_str), '-dpng');		
	disp(sprintf("done."));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1wk period-averaged freezing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tone_FSTI_1wk = mean(FSTI_1wk_tot(:,tone),2);
tone_FSTE_1wk = mean(FSTE_1wk_tot(:,tone),2);
tone_FSTC_1wk = mean(FSTC_1wk_tot(:,tone),2);
mean_tone_FSTI_1wk = mean(mean(FSTI_1wk_tot(:,tone),2));
mean_tone_FSTE_1wk = mean(mean(FSTE_1wk_tot(:,tone),2));
mean_tone_FSTC_1wk = mean(mean(FSTC_1wk_tot(:,tone),2));
%std_tone_FSTI_1wk = std(std(FSTI_1wk_tot(:,tone)))
%std_tone_FSTE_1wk = std(std(FSTE_1wk_tot(:,tone)))
%std_tone_FSTC_1wk = std(std(FSTC_1wk_tot(:,tone)))
std_tone_FSTI_1wk = std(mean(FSTI_1wk_tot(:,tone),2));
std_tone_FSTE_1wk = std(mean(FSTE_1wk_tot(:,tone),2));
std_tone_FSTC_1wk = std(mean(FSTC_1wk_tot(:,tone),2));

tone_post_tone_FSTI_1wk = mean(FSTI_1wk_tot(:,tone_post_tone),2);
tone_post_tone_FSTE_1wk = mean(FSTE_1wk_tot(:,tone_post_tone),2);
tone_post_tone_FSTC_1wk = mean(FSTC_1wk_tot(:,tone_post_tone),2);
mean_tone_post_tone_FSTI_1wk = mean(mean(FSTI_1wk_tot(:,tone_post_tone),2));
mean_tone_post_tone_FSTE_1wk = mean(mean(FSTE_1wk_tot(:,tone_post_tone),2));
mean_tone_post_tone_FSTC_1wk = mean(mean(FSTC_1wk_tot(:,tone_post_tone),2));
std_tone_post_tone_FSTI_1wk = std(mean(FSTI_1wk_tot(:,tone_post_tone),2));
std_tone_post_tone_FSTE_1wk = std(mean(FSTE_1wk_tot(:,tone_post_tone),2));
std_tone_post_tone_FSTC_1wk = std(mean(FSTC_1wk_tot(:,tone_post_tone),2));

post_tone_only_FSTI_1wk = mean(FSTI_1wk_tot(:,post_tone_only),2);
post_tone_only_FSTE_1wk = mean(FSTE_1wk_tot(:,post_tone_only),2);
post_tone_only_FSTC_1wk = mean(FSTC_1wk_tot(:,post_tone_only),2);
mean_post_tone_only_FSTI_1wk = mean(mean(FSTI_1wk_tot(:,post_tone_only),2));
mean_post_tone_only_FSTE_1wk = mean(mean(FSTE_1wk_tot(:,post_tone_only),2));
mean_post_tone_only_FSTC_1wk = mean(mean(FSTC_1wk_tot(:,post_tone_only),2));
std_post_tone_only_FSTI_1wk = std(mean(FSTI_1wk_tot(:,post_tone_only),2));
std_post_tone_only_FSTE_1wk = std(mean(FSTE_1wk_tot(:,post_tone_only),2));
std_post_tone_only_FSTC_1wk = std(mean(FSTC_1wk_tot(:,post_tone_only),2));

post_tone_20s_FSTI_1wk = mean(FSTI_1wk_tot(:,post_tone_20s),2);
post_tone_20s_FSTE_1wk = mean(FSTE_1wk_tot(:,post_tone_20s),2);
post_tone_20s_FSTC_1wk = mean(FSTC_1wk_tot(:,post_tone_20s),2);
mean_post_tone_20s_FSTI_1wk = mean(mean(FSTI_1wk_tot(:,post_tone_20s),2));
mean_post_tone_20s_FSTE_1wk = mean(mean(FSTE_1wk_tot(:,post_tone_20s),2));
mean_post_tone_20s_FSTC_1wk = mean(mean(FSTC_1wk_tot(:,post_tone_20s),2));
std_post_tone_20s_FSTI_1wk = std(mean(FSTI_1wk_tot(:,post_tone_20s),2));
std_post_tone_20s_FSTE_1wk = std(mean(FSTE_1wk_tot(:,post_tone_20s),2));
std_post_tone_20s_FSTC_1wk = std(mean(FSTC_1wk_tot(:,post_tone_20s),2));
sem_post_tone_20s_FSTI_1wk = std_post_tone_20s_FSTI_1wk./sqrt(size(FSTI_1wk_tot,1));
sem_post_tone_20s_FSTE_1wk = std_post_tone_20s_FSTE_1wk./sqrt(size(FSTE_1wk_tot,1));
sem_post_tone_20s_FSTC_1wk = std_post_tone_20s_FSTC_1wk./sqrt(size(FSTC_1wk_tot,1));

first_3min_FSTI_1wk = mean(FSTI_1wk_tot(:,first_3min),2);
first_3min_FSTE_1wk = mean(FSTE_1wk_tot(:,first_3min),2);
first_3min_FSTC_1wk = mean(FSTC_1wk_tot(:,first_3min),2);
mean_first_3min_FSTI_1wk = mean(mean(FSTI_1wk_tot(:,first_3min),2));
mean_first_3min_FSTE_1wk = mean(mean(FSTE_1wk_tot(:,first_3min),2));
mean_first_3min_FSTC_1wk = mean(mean(FSTC_1wk_tot(:,first_3min),2));
std_first_3min_FSTI_1wk = std(mean(FSTI_1wk_tot(:,first_3min),2));
std_first_3min_FSTE_1wk = std(mean(FSTE_1wk_tot(:,first_3min),2));
std_first_3min_FSTC_1wk = std(mean(FSTC_1wk_tot(:,first_3min),2));
sem_tone_FSTI_1wk = std_tone_FSTI_1wk./sqrt(size(FSTI_1wk_tot,1));
sem_tone_FSTE_1wk = std_tone_FSTE_1wk./sqrt(size(FSTE_1wk_tot,1));
sem_tone_FSTC_1wk = std_tone_FSTC_1wk./sqrt(size(FSTC_1wk_tot,1));
sem_tone_post_tone_FSTI_1wk = std_tone_post_tone_FSTI_1wk./sqrt(size(FSTI_1wk_tot,1));
sem_tone_post_tone_FSTE_1wk = std_tone_post_tone_FSTE_1wk./sqrt(size(FSTE_1wk_tot,1));
sem_tone_post_tone_FSTC_1wk = std_tone_post_tone_FSTC_1wk./sqrt(size(FSTC_1wk_tot,1));
sem_post_tone_only_FSTI_1wk = std_post_tone_only_FSTI_1wk./sqrt(size(FSTI_1wk_tot,1));
sem_post_tone_only_FSTE_1wk = std_post_tone_only_FSTE_1wk./sqrt(size(FSTE_1wk_tot,1));
sem_post_tone_only_FSTC_1wk = std_post_tone_only_FSTC_1wk./sqrt(size(FSTC_1wk_tot,1));
sem_first_3min_FSTI_1wk = std_first_3min_FSTI_1wk./sqrt(size(FSTI_1wk_tot,1));
sem_first_3min_FSTE_1wk = std_first_3min_FSTE_1wk./sqrt(size(FSTE_1wk_tot,1));
sem_first_3min_FSTC_1wk = std_first_3min_FSTC_1wk./sqrt(size(FSTC_1wk_tot,1));

% The plot

%% OBSOLETE

figure; hold on;
bar(1,mean_tone_FSTE_1wk,'r');
bar(2,mean_tone_FSTI_1wk,'b');
bar(3,mean_tone_FSTC_1wk,'k');
bar(4,mean_tone_post_tone_FSTE_1wk,'r');
bar(5,mean_tone_post_tone_FSTI_1wk,'b');
bar(6,mean_tone_post_tone_FSTC_1wk,'k');
bar(7,mean_post_tone_only_FSTE_1wk,'r');
bar(8,mean_post_tone_only_FSTI_1wk,'b');
bar(9,mean_post_tone_only_FSTC_1wk,'k');
bar(10,mean_post_tone_20s_FSTI_1wk,'b');
bar(11,mean_post_tone_20s_FSTE_1wk,'r');
bar(12,mean_post_tone_20s_FSTC_1wk,'k');
set(gca,'XTick',[1:12]);
set(gca,'XTickLabels',{'Tone','Tone','Tone','Tone+post-tone','Tone+post-tone','Tone+post-tone','Post-tone only','Post-tone only','Post-tone only','Post-tone 20s','Post-tone 20s','Post-tone 20s'});
set(gca,'XTickLabelRotation',-45);
legend({'SST Exc','SST Inh','SST Ctl'});
title('Test B (1 week)');
ylabel('Freezing (%)');

%% /OBSOLETE

% The plot - paper

f=figure; hold on;
%use_test = 'anova'
%use_test = 'ranksum'
%use_test = 'kw';

%post_hoc = 'hsd'
%post_hoc = 'bonferroni'
%post_hoc = 'dunn-sidak'

if for_paper
	f.PaperPosition = [0 0 2 2];
else
	f.PaperPosition = [0 0 4 4];
end
file_str = '5a_1wk_period_freezing';
%set(gca,'FontSize',18);

data_E = mean(FSTE_1wk_tot(:,tone),2);
data_I = mean(FSTI_1wk_tot(:,tone),2);
data_C = mean(FSTC_1wk_tot(:,tone),2);

%{
data_E = reshape(FSTE_1wk_tot(:,tone), 1,[]);
data_I = reshape(FSTI_1wk_tot(:,tone), 1,[]);
data_C = reshape(FSTC_1wk_tot(:,tone), 1,[]);
%}

[G, P] = do_stats_tfc(data_E, data_I, data_C, use_test, post_hoc, 0);

b = bar([1 2 3], [mean_tone_FSTE_1wk; mean_tone_FSTI_1wk; mean_tone_FSTC_1wk], 1.0); b.FaceColor='flat'; b.CData(1,:)=my_r; b.CData(2,:)=my_b, b.CData(3,:)=my_k; 
max_sem = max([sem_tone_FSTE_1wk, sem_tone_FSTI_1wk, sem_tone_FSTC_1wk]);
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',14,'nosort',1,'max_errbar',max_sem));
e = errorbar(1, mean_tone_FSTE_1wk, sem_tone_FSTE_1wk, 'Color',my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(2, mean_tone_FSTI_1wk, sem_tone_FSTI_1wk, 'Color',my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(3, mean_tone_FSTC_1wk, sem_tone_FSTC_1wk, 'Color',my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
%sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',14,'nosort',1,'max_errbar',max_sem));

data_E = mean(FSTE_1wk_tot(:,post_tone_only),2);
data_I = mean(FSTI_1wk_tot(:,post_tone_only),2);
data_C = mean(FSTC_1wk_tot(:,post_tone_only),2);

%{
data_E = mean(FSTE_1wk_tot(:,tone_post_tone),2);
data_I = mean(FSTI_1wk_tot(:,tone_post_tone),2);
data_C = mean(FSTC_1wk_tot(:,tone_post_tone),2);
%}

%{
data_E = reshape(FSTE_1wk_tot(:,post_tone_only), 1,[]);
data_I = reshape(FSTI_1wk_tot(:,post_tone_only), 1,[]);
data_C = reshape(FSTC_1wk_tot(:,post_tone_only), 1,[]);
%}

[G, P] = do_stats_tfc(data_E, data_I, data_C, use_test, post_hoc, 4);

b = bar([5 6 7], [mean_post_tone_only_FSTE_1wk; mean_post_tone_only_FSTI_1wk; mean_post_tone_only_FSTC_1wk], 1.0); b.FaceColor='flat'; b.CData(1,:)=my_r; b.CData(2,:)=my_b, b.CData(3,:)=my_k; 
max_sem = max([sem_post_tone_only_FSTE_1wk, sem_post_tone_only_FSTI_1wk, sem_post_tone_only_FSTC_1wk]);
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',14,'nosort',1,'max_errbar',max_sem));

e = errorbar(5, mean_post_tone_only_FSTE_1wk, sem_post_tone_only_FSTE_1wk, 'Color',my_r); e.CapSize=0; e.LineWidth=2;
e = errorbar(6, mean_post_tone_only_FSTI_1wk, sem_post_tone_only_FSTI_1wk, 'Color',my_b); e.CapSize=0; e.LineWidth=2;
e = errorbar(7, mean_post_tone_only_FSTC_1wk, sem_post_tone_only_FSTC_1wk, 'Color',my_k); e.CapSize=0; e.LineWidth=2;

set(gca,'XTick',[1:7]);
%set(gca,'XTickLabels',{'Tone','Tone','Tone', '', 'Post tone','Post tone','Post tone'});
%set(gca,'XTickLabels',{'Exc-Tone','Inh-Tone','Ctl-Tone', '', 'Exc-ITI','Inh-ITI','Ctl-ITI'});
set(gca,'XTickLabels',{'Exc          ','Inh     ','Ctl     ', '', 'Exc     ','Inh     ','Ctl     '});
set(gca,'XTickLabelRotation',-45);

ylabel('Freezing (%)');
title('Test 1wk');

if saveit
	print(f, sprintf('%s\\%s.svg',file_path,file_str), '-dsvg');
	print(f, sprintf('%s\\%s.png',file_path,file_str), '-dpng');		
	disp(sprintf("done."));
end

%%
%% 1wk period-averaged freezing - but with first_3min
%%

% The plot - paper
f=figure; hold on;
%use_test = 'anova'
%use_test = 'ranksum'
%use_test = 'kw';
%post_hoc = 'hsd'
%post_hoc = 'bonferroni'
%post_hoc = 'dunn-sidak'
%use_test = 'ranksum'

if for_paper
	f.PaperPosition = [0 0 3 2];
else
	f.PaperPosition = [0 0 6 4];
end
%set(gca,'FontSize',18);
file_str = '5b_1wk_period_freezing_first_3min';

%{
[p_EI, h_EI] = ranksum(mean(FSTE_1wk_tot(:,tone),2), mean(FSTI_1wk_tot(:,tone),2));
[p_EC, h_EC] = ranksum(mean(FSTE_1wk_tot(:,tone),2), mean(FSTC_1wk_tot(:,tone),2));
[p_IC, h_IC] = ranksum(mean(FSTI_1wk_tot(:,tone),2), mean(FSTC_1wk_tot(:,tone),2));
G = {[2 3], [1 2], [1 3]};
P = [p_IC, p_EI, p_EC];
%}

data_E = mean(FSTE_1wk_tot(:,tone),2);
data_I = mean(FSTI_1wk_tot(:,tone),2);
data_C = mean(FSTC_1wk_tot(:,tone),2);
[G, P] = do_stats_tfc(data_E, data_I, data_C, use_test, post_hoc, 0);

b = bar([1 2 3], [mean_tone_FSTE_1wk; mean_tone_FSTI_1wk; mean_tone_FSTC_1wk], 1.0); b.FaceColor='flat'; b.CData(1,:)=my_r; b.CData(2,:)=my_b, b.CData(3,:)=my_k; 
max_sem = max([sem_tone_FSTE_1wk, sem_tone_FSTI_1wk, sem_tone_FSTC_1wk]);
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',8,'nosort',1,'max_errbar',max_sem));
%sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',14,'nosort',1,'max_errbar',max_sem));
e = errorbar(1, mean_tone_FSTE_1wk, sem_tone_FSTE_1wk, 'Color',my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(2, mean_tone_FSTI_1wk, sem_tone_FSTI_1wk, 'Color',my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(3, mean_tone_FSTC_1wk, sem_tone_FSTC_1wk, 'Color',my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;

%{
[p_EI, h_EI] = ranksum(mean(FSTE_1wk_tot(:,post_tone_only),2), mean(FSTI_1wk_tot(:,post_tone_only),2));
[p_EC, h_EC] = ranksum(mean(FSTE_1wk_tot(:,post_tone_only),2), mean(FSTC_1wk_tot(:,post_tone_only),2));
[p_IC, h_IC] = ranksum(mean(FSTI_1wk_tot(:,post_tone_only),2), mean(FSTC_1wk_tot(:,post_tone_only),2));
G = {[6 7], [5 6], [5 7]};
P = [p_IC, p_EI, p_EC];
%}

data_E = mean(FSTE_1wk_tot(:,post_tone_only),2);
data_I = mean(FSTI_1wk_tot(:,post_tone_only),2);
data_C = mean(FSTC_1wk_tot(:,post_tone_only),2);
[G, P] = do_stats_tfc(data_E, data_I, data_C, use_test, post_hoc, 4);

b = bar([5 6 7], [mean_post_tone_only_FSTE_1wk; mean_post_tone_only_FSTI_1wk; mean_post_tone_only_FSTC_1wk], 1.0); b.FaceColor='flat'; b.CData(1,:)=my_r; b.CData(2,:)=my_b, b.CData(3,:)=my_k; 
max_sem = max([sem_post_tone_only_FSTE_1wk, sem_post_tone_only_FSTI_1wk, sem_post_tone_only_FSTC_1wk]);
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',14,'nosort',1,'max_errbar',max_sem));
e = errorbar(5, mean_post_tone_only_FSTE_1wk, sem_post_tone_only_FSTE_1wk, 'Color',my_r); e.CapSize=0; e.LineWidth=2;
e = errorbar(6, mean_post_tone_only_FSTI_1wk, sem_post_tone_only_FSTI_1wk, 'Color',my_b); e.CapSize=0; e.LineWidth=2;
e = errorbar(7, mean_post_tone_only_FSTC_1wk, sem_post_tone_only_FSTC_1wk, 'Color',my_k); e.CapSize=0; e.LineWidth=2;

%{
[p_EI, h_EI] = ranksum(mean(FSTE_1wk_tot(:,post_tone_only),2), mean(FSTI_1wk_tot(:,post_tone_only),2));
[p_EC, h_EC] = ranksum(mean(FSTE_1wk_tot(:,post_tone_only),2), mean(FSTC_1wk_tot(:,post_tone_only),2));
[p_IC, h_IC] = ranksum(mean(FSTI_1wk_tot(:,post_tone_only),2), mean(FSTC_1wk_tot(:,post_tone_only),2));
G = {[10 11], [9 10], [9 11]};
P = [p_IC, p_EI, p_EC];
%}

data_E = mean(FSTE_1wk_tot(:,first_3min),2);
data_I = mean(FSTI_1wk_tot(:,first_3min),2);
data_C = mean(FSTC_1wk_tot(:,first_3min),2);
[G, P] = do_stats_tfc(data_E, data_I, data_C, use_test, post_hoc, 8);

b = bar([9 10 11], [mean_first_3min_FSTE_1wk; mean_first_3min_FSTI_1wk; mean_first_3min_FSTC_1wk], 1.0); b.FaceColor='flat'; b.CData(1,:)=my_r; b.CData(2,:)=my_b, b.CData(3,:)=my_k; 
max_sem = max([sem_first_3min_FSTE_1wk, sem_first_3min_FSTI_1wk, sem_first_3min_FSTC_1wk]);
%sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',14,'nosort',1,'max_errbar',max_sem));
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',8,'nosort',1,'max_errbar',max_sem));
%e = errorbar([1 2 3], [mean_tone_FSTE; mean_tone_FSTI; mean_tone_FSTC], [std_tone_FSTE; std_tone_FSTI; std_tone_FSTC], 'k.'); e.CapSize=0; e.LineWidth=2;
e = errorbar(9, mean_first_3min_FSTE_1wk, sem_first_3min_FSTE_1wk, 'Color',my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(10, mean_first_3min_FSTI_1wk, sem_first_3min_FSTI_1wk, 'Color',my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(11, mean_first_3min_FSTC_1wk, sem_first_3min_FSTC_1wk, 'Color',my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;

set(gca,'XTick',[1:11]);
set(gca,'XTickLabels',{'Tone','Tone','Tone', '', 'Post tone','Post tone','Post tone','',strcat('First ', first_min),strcat('First ', first_min), strcat('First ', first_min)});
set(gca,'XTickLabelRotation',-45);

title('Test 1wk');

if saveit
	print(f, sprintf('%s\\%s.svg',file_path,file_str), '-dsvg');
	print(f, sprintf('%s\\%s.png',file_path,file_str), '-dpng');		
	disp(sprintf("done."));
end


% Compare across periods (Test B 1wk)

f=figure; hold on;
if for_paper
	f.PaperPosition = [0 0 3 2];
else
	f.PaperPosition = [0 0 6 4];
end
%set(gca,'FontSize',18);
file_str = '5c_1wk_period_freezing_first_3min_comparison';

b = bar([1 2 3], [mean_first_3min_FSTE_1wk; mean_tone_FSTE_1wk; mean_post_tone_only_FSTE_1wk], 1.0); b.FaceColor='flat'; b.CData(1,:)=my_r; b.CData(2,:)=my_r, b.CData(3,:)=my_r;
e = errorbar(1, mean_first_3min_FSTE_1wk, sem_first_3min_FSTE_1wk, 'Color',my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(2, mean_tone_FSTE_1wk, sem_tone_FSTE_1wk, 'Color',my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(3, mean_post_tone_only_FSTE_1wk, sem_post_tone_only_FSTE_1wk, 'Color',my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
scatter(ones(1,length(first_3min_FSTE_1wk)), first_3min_FSTE_1wk, my_sz,my_k);
scatter(ones(1,length(tone_FSTE_1wk))+1, tone_FSTE_1wk, my_sz,my_k);
scatter(ones(1,length(post_tone_only_FSTE_1wk))+2, post_tone_only_FSTE_1wk, my_sz,my_k);
for i=1:length(first_3min_FSTE_1wk)
	plot([1 2], [first_3min_FSTE_1wk(i), tone_FSTE_1wk(i)], 'k-');
	plot([2 3], [tone_FSTE_1wk(i), post_tone_only_FSTE_1wk(i)], 'k-');
end

max_sem = max([sem_first_3min_FSTE_1wk, sem_tone_FSTE_1wk, sem_post_tone_only_FSTE_1wk]);
data = {mean(FSTE_1wk_tot(:,first_3min),2), mean(FSTE_1wk_tot(:,tone),2), mean(FSTE_1wk_tot(:,post_tone_only),2)};
[G, P] = stats_ranova(data, post_hoc, 0);
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.07,'FontSize',14,'nosort',1,'max_errbar',max_sem));

b = bar([5 6 7], [mean_first_3min_FSTI_1wk; mean_tone_FSTI_1wk; mean_post_tone_only_FSTI_1wk], 1.0); b.FaceColor='flat'; b.CData(1,:)=my_b; b.CData(2,:)=my_b, b.CData(3,:)=my_b;
e = errorbar(5, mean_first_3min_FSTI_1wk, sem_first_3min_FSTI_1wk, 'Color',my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(6, mean_tone_FSTI_1wk, sem_tone_FSTI_1wk, 'Color',my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(7, mean_post_tone_only_FSTI_1wk, sem_post_tone_only_FSTI_1wk, 'Color',my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
scatter(ones(1,length(first_3min_FSTI_1wk))+4, first_3min_FSTI_1wk, my_sz,my_k);
scatter(ones(1,length(tone_FSTI_1wk))+5, tone_FSTI_1wk, my_sz,my_k);
scatter(ones(1,length(post_tone_only_FSTI_1wk))+6, post_tone_only_FSTI_1wk, my_sz,my_k);
for i=1:length(first_3min_FSTI_1wk)
	plot([5 6], [first_3min_FSTI_1wk(i), tone_FSTI_1wk(i)], 'k-');
	plot([6 7], [tone_FSTI_1wk(i), post_tone_only_FSTI_1wk(i)], 'k-');
end

max_sem = max([sem_first_3min_FSTE_1wk, sem_tone_FSTE_1wk, sem_post_tone_only_FSTE_1wk]);
data = {mean(FSTI_1wk_tot(:,first_3min),2), mean(FSTI_1wk_tot(:,tone),2), mean(FSTI_1wk_tot(:,post_tone_only),2)};
[G, P] = stats_ranova(data, post_hoc, 4);
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.07,'FontSize',14,'nosort',1,'max_errbar',max_sem));

b = bar([9 10 11], [mean_first_3min_FSTC_1wk; mean_tone_FSTC_1wk; mean_post_tone_only_FSTC_1wk], 1.0); b.FaceColor='flat'; b.CData(1,:)=my_k; b.CData(2,:)=my_k, b.CData(3,:)=my_k;
e = errorbar(9, mean_first_3min_FSTC_1wk, sem_first_3min_FSTC_1wk, 'Color',my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(10, mean_tone_FSTC_1wk, sem_tone_FSTC_1wk, 'Color',my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(11, mean_post_tone_only_FSTC_1wk, sem_post_tone_only_FSTC_1wk, 'Color',my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
scatter(ones(1,length(first_3min_FSTC_1wk))+8, first_3min_FSTC_1wk, my_sz,my_h);
scatter(ones(1,length(tone_FSTC_1wk))+9, tone_FSTC_1wk, my_sz,my_h);
scatter(ones(1,length(post_tone_only_FSTC_1wk))+10, post_tone_only_FSTC_1wk, my_sz,my_h);
for i=1:length(first_3min_FSTC_1wk)
	plot([9 10], [first_3min_FSTC_1wk(i), tone_FSTC_1wk(i)], 'Color',my_h);
	plot([10 11], [tone_FSTC_1wk(i), post_tone_only_FSTC_1wk(i)], 'Color',my_h);
end

max_sem = max([sem_first_3min_FSTC_1wk, sem_tone_FSTC_1wk, sem_post_tone_only_FSTC_1wk]);
data = {mean(FSTC_1wk_tot(:,first_3min),2), mean(FSTC_1wk_tot(:,tone),2), mean(FSTC_1wk_tot(:,post_tone_only),2)};
[G, P] = stats_ranova(data, post_hoc, 8);
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.07,'FontSize',14,'nosort',1,'max_errbar',max_sem));

set(gca,'XTick',[1:11]);
set(gca,'XTickLabels',{strcat('First ', first_min),'Tone','Post tone', '', strcat('First ', first_min),'Tone','Post tone', '', strcat('First ', first_min),'Tone','Post tone'});
set(gca,'XTickLabelRotation',-45);
ylabel('Freezing (%)');
title('Test B 1wk - within groups');

if saveit
	print(f, sprintf('%s\\%s.svg',file_path,file_str), '-dsvg');
	print(f, sprintf('%s\\%s.png',file_path,file_str), '-dpng');		
	disp(sprintf("done."));
end

%%
%% Compare 48hr vs 1wk
%%

f=figure; hold on;
if for_paper
	f.PaperPosition = [0 0 4 2];
else
	f.PaperPosition = [0 0 8 4];
end
%set(gca,'FontSize',18);
file_str = '5d_compare_48hr_1wk_period_freezing';

b = bar([1 2 3 4 5 6], [mean_tone_FSTE; mean_tone_FSTE_1wk; mean_tone_FSTI; mean_tone_FSTI_1wk; mean_tone_FSTC; mean_tone_FSTC_1wk], 1.0); 
%b.FaceColor='flat'; b.CData(1,:)=[1 0 0]; b.CData(2,:)=[1 0.5 0.5]; b.CData(3,:)=[0 0 1]; b.CData(4,:)=[0.5 0.5 1]; b.CData(5,:)=[0 0 0]; b.CData(6,:)=[0.5 0.5 0.5]; 
%b.FaceColor='flat'; b.CData(1,:)=[1 0 0]; b.CData(2,:)=[1 0 0]; b.CData(3,:)=[0 0 1]; b.CData(4,:)=[0 0 1]; b.CData(5,:)=[0 0 0]; b.CData(6,:)=[0 0 0]; 
b.FaceColor='flat'; b.CData(1,:)=my_r; b.CData(2,:)=my_r; b.CData(3,:)=my_b; b.CData(4,:)=my_b; b.CData(5,:)=my_k; b.CData(6,:)=my_k; 
e = errorbar(1, mean_tone_FSTE, sem_tone_FSTE, 'Color',my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(2, mean_tone_FSTE_1wk, sem_tone_FSTE_1wk, 'Color',my_r); e.CapSize=0; e.LineWidth=2; %'.', 'Color',[1 0.5 0.5]); e.CapSize=0; e.LineWidth=2;
e = errorbar(3, mean_tone_FSTI, sem_tone_FSTI, 'Color',my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(4, mean_tone_FSTI_1wk, sem_tone_FSTI_1wk, 'Color',my_b); e.CapSize=0; e.LineWidth=2; %'.', 'Color',[0.5 0.5 1]); e.CapSize=0; e.LineWidth=2;
e = errorbar(5, mean_tone_FSTC, sem_tone_FSTC, 'Color',my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(6, mean_tone_FSTC_1wk, sem_tone_FSTC_1wk, 'Color',my_k); e.CapSize=0; e.LineWidth=2; %'.', 'Color',[0.5 0.5 0.5]); e.CapSize=0; e.LineWidth=2;

%{
scatter(ones(1,length(tone_FSTE)), tone_FSTE, my_sz,my_k);
scatter(ones(1,length(tone_FSTE_1wk))+1, tone_FSTE_1wk, my_sz,my_k);
for i=1:length(tone_FSTE)
	plot([1 2], [tone_FSTE(i), tone_FSTE_1wk(i)], 'Color',my_k);
end
scatter(ones(1,length(tone_FSTI))+2, tone_FSTI, my_sz,my_k);
scatter(ones(1,length(tone_FSTI_1wk))+3, tone_FSTI_1wk, my_sz,my_k);
for i=1:length(tone_FSTI)
	plot([3 4], [tone_FSTI(i), tone_FSTI_1wk(i)], 'Color',my_k);
end
scatter(ones(1,length(tone_FSTC))+4, tone_FSTC, my_sz,my_h);
scatter(ones(1,length(tone_FSTC_1wk))+5, tone_FSTC_1wk, my_sz,my_h);
for i=1:length(tone_FSTC)
	plot([5 6], [tone_FSTC(i), tone_FSTC_1wk(i)], 'Color',my_h);
end
%}

max_sem = max([sem_tone_FSTE, sem_tone_FSTE_1wk; sem_tone_FSTI, sem_tone_FSTI_1wk; sem_tone_FSTC, sem_tone_FSTC_1wk]);
[h_dwE, p_dwE, ci, stats] = ttest(mean(FSTE_tot(:,tone),2), mean(FSTE_1wk_tot(:,tone),2));
%[p_dwE, h_dwE] = ranksum(mean(FSTE_tot(:,tone),2), mean(FSTE_1wk_tot(:,tone),2));
[h_dwI, p_dwI, ci, stats] = ttest(mean(FSTI_tot(:,tone),2), mean(FSTI_1wk_tot(:,tone),2));
[h_dwC, p_dwC, ci, stats] = ttest(mean(FSTC_tot(:,tone),2), mean(FSTC_1wk_tot(:,tone),2));

G = {[1 2], [3 4], [5 6]};
P = [p_dwE, p_dwI, p_dwC];
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',14,'nosort',1,'max_errbar',max_sem));

b = bar([8 9 10 11 12 13], [mean_post_tone_only_FSTE; mean_post_tone_only_FSTE_1wk; mean_post_tone_only_FSTI; mean_post_tone_only_FSTI_1wk; mean_post_tone_only_FSTC; mean_post_tone_only_FSTC_1wk], 1.0); 
%b.FaceColor='flat'; b.CData(1,:)=[1 0 0]; b.CData(2,:)=[1 0.5 0.5]; b.CData(3,:)=[0 0 1]; b.CData(4,:)=[0.5 0.5 1]; b.CData(5,:)=[0 0 0]; b.CData(6,:)=[0.5 0.5 0.5]; 
%b.FaceColor='flat'; b.CData(1,:)=[1 0 0]; b.CData(2,:)=[1 0 0]; b.CData(3,:)=[0 0 1]; b.CData(4,:)=[0 0 1]; b.CData(5,:)=[0 0 0]; b.CData(6,:)=[0 0 0]; 
b.FaceColor='flat'; b.CData(1,:)=my_r; b.CData(2,:)=my_r; b.CData(3,:)=my_b; b.CData(4,:)=my_b; b.CData(5,:)=my_k; b.CData(6,:)=my_k; 
e = errorbar(8, mean_post_tone_only_FSTE, sem_post_tone_only_FSTE, 'Color',my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(9, mean_post_tone_only_FSTE_1wk, sem_post_tone_only_FSTE_1wk, 'Color',my_r); e.CapSize=0; e.LineWidth=2; %'.', 'Color',[1 0.5 0.5]); e.CapSize=0; e.LineWidth=2;
e = errorbar(10, mean_post_tone_only_FSTI, sem_post_tone_only_FSTI, 'Color',my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(11, mean_post_tone_only_FSTI_1wk, sem_post_tone_only_FSTI_1wk, 'Color',my_b); e.CapSize=0; e.LineWidth=2; %'.', 'Color',[0.5 0.5 1]); e.CapSize=0; e.LineWidth=2;
e = errorbar(12, mean_post_tone_only_FSTC, sem_post_tone_only_FSTC, 'Color',my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(13, mean_post_tone_only_FSTC_1wk, sem_post_tone_only_FSTC_1wk, 'Color',my_k); e.CapSize=0; e.LineWidth=2; %'.', 'Color',[0.5 0.5 0.5]); e.CapSize=0; e.LineWidth=2;

%{
scatter(ones(1,length(post_tone_only_FSTE))+7, post_tone_only_FSTE, my_sz,my_k);
scatter(ones(1,length(post_tone_only_FSTE_1wk))+8, post_tone_only_FSTE_1wk, my_sz,my_k);
for i=1:length(post_tone_only_FSTE)
	plot([8 9], [post_tone_only_FSTE(i), post_tone_only_FSTE_1wk(i)], 'Color',my_k);
end
scatter(ones(1,length(post_tone_only_FSTI))+9, post_tone_only_FSTI, my_sz,my_k);
scatter(ones(1,length(post_tone_only_FSTI_1wk))+10, post_tone_only_FSTI_1wk, my_sz,my_k);
for i=1:length(post_tone_only_FSTI)
	plot([10 11], [post_tone_only_FSTI(i), post_tone_only_FSTI_1wk(i)], 'Color',my_k);
end
scatter(ones(1,length(post_tone_only_FSTC))+11, post_tone_only_FSTC, my_sz,my_h);
scatter(ones(1,length(post_tone_only_FSTC_1wk))+12, post_tone_only_FSTC_1wk, my_sz,my_h);
for i=1:length(post_tone_only_FSTC)
	plot([12 13], [post_tone_only_FSTC(i), post_tone_only_FSTC_1wk(i)], 'Color',my_h);
end
%}

max_sem = max([sem_post_tone_only_FSTE, sem_post_tone_only_FSTE_1wk; sem_post_tone_only_FSTI, sem_post_tone_only_FSTI_1wk; sem_post_tone_only_FSTC, sem_post_tone_only_FSTC_1wk]);
[h_dwE, p_dwE, ci, stats] = ttest(mean(FSTE_tot(:,post_tone_only),2), mean(FSTE_1wk_tot(:,post_tone_only),2));
[h_dwI, p_dwI, ci, stats] = ttest(mean(FSTI_tot(:,post_tone_only),2), mean(FSTI_1wk_tot(:,post_tone_only),2));
[h_dwC, p_dwC, ci, stats] = ttest(mean(FSTC_tot(:,post_tone_only),2), mean(FSTC_1wk_tot(:,post_tone_only),2));

G = {[8 9], [10 11], [12 13]};
P = [p_dwE, p_dwI, p_dwC];
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',14,'nosort',1,'max_errbar',max_sem));

b = bar([15 16 17 18 19 20], [mean_first_3min_FSTE; mean_first_3min_FSTE_1wk; mean_first_3min_FSTI; mean_first_3min_FSTI_1wk; mean_first_3min_FSTC; mean_first_3min_FSTC_1wk], 1.0); 
%b.FaceColor='flat'; b.CData(1,:)=[1 0 0]; b.CData(2,:)=[1 0.5 0.5]; b.CData(3,:)=[0 0 1]; b.CData(4,:)=[0.5 0.5 1]; b.CData(5,:)=[0 0 0]; b.CData(6,:)=[0.5 0.5 0.5]; 
%b.FaceColor='flat'; b.CData(1,:)=[1 0 0]; b.CData(2,:)=[1 0 0]; b.CData(3,:)=[0 0 1]; b.CData(4,:)=[0 0 1]; b.CData(5,:)=[0 0 0]; b.CData(6,:)=[0 0 0]; 
b.FaceColor='flat'; b.CData(1,:)=my_r; b.CData(2,:)=my_r; b.CData(3,:)=my_b; b.CData(4,:)=my_b; b.CData(5,:)=my_k; b.CData(6,:)=my_k; 
e = errorbar(15, mean_first_3min_FSTE, sem_first_3min_FSTE, 'Color',my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(16, mean_first_3min_FSTE_1wk, sem_first_3min_FSTE_1wk, 'Color',my_r); e.CapSize=0; e.LineWidth=2; %'.', 'Color',[1 0.5 0.5]); e.CapSize=0; e.LineWidth=2;
e = errorbar(17, mean_first_3min_FSTI, sem_first_3min_FSTI, 'Color',my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(18, mean_first_3min_FSTI_1wk, sem_first_3min_FSTI_1wk, 'Color',my_b); e.CapSize=0; e.LineWidth=2; %'.', 'Color',[0.5 0.5 1]); e.CapSize=0; e.LineWidth=2;
e = errorbar(19, mean_first_3min_FSTC, sem_first_3min_FSTC, 'Color',my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(20, mean_first_3min_FSTC_1wk, sem_first_3min_FSTC_1wk, 'Color',my_k); e.CapSize=0; e.LineWidth=2; %'.', 'Color',[0.5 0.5 0.5]); e.CapSize=0; e.LineWidth=2;

%{
scatter(ones(1,length(post_tone_only_FSTE))+14, post_tone_only_FSTE, my_sz,my_k);
scatter(ones(1,length(post_tone_only_FSTE_1wk))+15, post_tone_only_FSTE_1wk, my_sz,my_k);
for i=1:length(post_tone_only_FSTE)
	plot([15 16], [post_tone_only_FSTE(i), post_tone_only_FSTE_1wk(i)], 'Color',my_k);
end
scatter(ones(1,length(post_tone_only_FSTI))+16, post_tone_only_FSTI, my_sz,my_k);
scatter(ones(1,length(post_tone_only_FSTI_1wk))+17, post_tone_only_FSTI_1wk, my_sz,my_k);
for i=1:length(post_tone_only_FSTI)
	plot([17 18], [post_tone_only_FSTI(i), post_tone_only_FSTI_1wk(i)], 'Color',my_k);
end
scatter(ones(1,length(post_tone_only_FSTC))+18, post_tone_only_FSTC, my_sz,my_h);
scatter(ones(1,length(post_tone_only_FSTC_1wk))+19, post_tone_only_FSTC_1wk, my_sz,my_h);
for i=1:length(post_tone_only_FSTC)
	plot([19 20], [post_tone_only_FSTC(i), post_tone_only_FSTC_1wk(i)], 'Color',my_h);
end
%}

max_sem = max([sem_first_3min_FSTE, sem_first_3min_FSTE_1wk; sem_first_3min_FSTI, sem_first_3min_FSTI_1wk; sem_first_3min_FSTC, sem_first_3min_FSTC_1wk]);
[h_dwE, p_dwE, ci, stats] = ttest(mean(FSTE_tot(:,first_3min),2), mean(FSTE_1wk_tot(:,first_3min),2));
[h_dwI, p_dwI, ci, stats] = ttest(mean(FSTI_tot(:,first_3min),2), mean(FSTI_1wk_tot(:,first_3min),2));
[h_dwC, p_dwC, ci, stats] = ttest(mean(FSTC_tot(:,first_3min),2), mean(FSTC_1wk_tot(:,first_3min),2));

G = {[15 16], [17 18], [19 20]};
P = [p_dwE, p_dwI, p_dwC];
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',14,'nosort',1,'max_errbar',max_sem));

set(gca,'XTick',[1:20]);
%set(gca,'XTickLabels',{'Tone 48hr', 'Tone 1wk', 'Tone 48hr', 'Tone 1wk', 'Tone 48hr', 'Tone 1wk', '', 'Post tone 48hr', 'Post tone 1wk', 'Post tone 48hr', 'Post tone 1wk', 'Post tone 48hr', 'Post tone 1wk', '', strcat('First',first_min,' 48hr'), strcat('First',first_min,' 1wk'), strcat('First',first_min,' 48hr'), strcat('First',first_min,' 1wk'), strcat('First',first_min,' 48hr'), strcat('First',first_min,' 1wk')});
%set(gca,'XTickLabels',{'48hr', '1wk', '48hr', '1wk', '48hr', '1wk', '', '48hr', '1wk', '48hr', '1wk', '48hr', '1wk', '', '48hr', '1wk','48hr','1wk','48hr','1wk'});
%set(gca,'XTickLabels',{'Exc-48hr', 'Exc-1wk ', 'Inh-48hr', 'Inh-1wk ', 'Ctl-48hr', 'Ctl-1wk ', '', 'Exc-48hr', 'Exc-1wk ', 'Inh-48hr', 'Inh-1wk ', 'Ctl-48hr', 'Ctl-1wk ', '', 'Exc-48hr', 'Exc-1wk ','Inh-48hr','Inh-1wk ','Ctl-48hr','Ctl-1wk '});
set(gca,'XTickLabels',{'Exc          ','Exc+','Inh','Inh+','Ctl','Ctl+','','Exc','Exc+','Inh','Inh+','Ctl','Ctl+','','Exc','Exc+','Inh','Inh+','Ctl','Ctl+'});
set(gca,'XTickLabelRotation',-45);

ylabel('Freezing (%)');
title('Test 48hr vs 1wk');

if saveit
	print(f, sprintf('%s\\%s.svg',file_path,file_str), '-dsvg');
	print(f, sprintf('%s\\%s.png',file_path,file_str), '-dpng');		
	disp(sprintf("done."));
end

%
% Compare first 3min 48hr and 1wk
%

f=figure; hold on;
if for_paper
	f.PaperPosition = [0 0 2 2];
else
	f.PaperPosition = [0 0 4 4];
end
%set(gca,'FontSize',18);
file_str = '5e_compare_48hr_1wk_first_3min';

b = bar([1 2 4 5 7 8], [mean_first_3min_FSTE; mean_first_3min_FSTE_1wk; mean_first_3min_FSTI; mean_first_3min_FSTI_1wk; mean_first_3min_FSTC; mean_first_3min_FSTC_1wk], 1.0); 
%b.FaceColor='flat'; b.CData(1,:)=[1 0 0]; b.CData(2,:)=[1 0 0]; b.CData(3,:)=[0 0 1]; b.CData(4,:)=[0 0 1]; b.CData(5,:)=[0 0 0]; b.CData(6,:)=[0 0 0]; 
b.FaceColor='flat'; b.CData(1,:)=my_r; b.CData(2,:)=my_r; b.CData(3,:)=my_b; b.CData(4,:)=my_b; b.CData(5,:)=my_k; b.CData(6,:)=my_k; 
e = errorbar(1, mean_first_3min_FSTE, sem_first_3min_FSTE, 'Color',my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(2, mean_first_3min_FSTE_1wk, sem_first_3min_FSTE_1wk, 'Color',my_r); e.CapSize=0; e.LineWidth=2; %'.', 'Color',[1 0.5 0.5]); e.CapSize=0; e.LineWidth=2;
e = errorbar(4, mean_first_3min_FSTI, sem_first_3min_FSTI, 'Color',my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(5, mean_first_3min_FSTI_1wk, sem_first_3min_FSTI_1wk, 'Color',my_b); e.CapSize=0; e.LineWidth=2; %'.', 'Color',[0.5 0.5 1]); e.CapSize=0; e.LineWidth=2;
e = errorbar(7, mean_first_3min_FSTC, sem_first_3min_FSTC, 'Color',my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(8, mean_first_3min_FSTC_1wk, sem_first_3min_FSTC_1wk, 'Color',my_k); e.CapSize=0; e.LineWidth=2; %'.', 'Color',[0.5 0.5 0.5]); e.CapSize=0; e.LineWidth=2;

%{
X=mean(FSTE_tot(:,first_3min),2);
Y=mean(FSTE_1wk_tot(:,first_3min),2);
scatter(ones(1,length(X)), X, 'k');
scatter(ones(1,length(Y))+1, Y, 'k');
for i=1:length(X)
	plot([1 2], [X(i), Y(i)], 'Color','k');
end

X=mean(FSTI_tot(:,first_3min),2);
Y=mean(FSTI_1wk_tot(:,first_3min),2);
scatter(ones(1,length(X))+3, X, 'k');
scatter(ones(1,length(Y))+4, Y, 'k');
for i=1:length(X)
	plot([4 5], [X(i), Y(i)], 'Color','k');
end

X=mean(FSTC_tot(:,first_3min),2);
Y=mean(FSTC_1wk_tot(:,first_3min),2);
scatter(ones(1,length(X))+6, X, [],my_h);
scatter(ones(1,length(Y))+7, Y, [], my_h);
for i=1:length(X)
	plot([7 8], [X(i), Y(i)], 'Color',my_h);
end
%}

max_sem = max([sem_first_3min_FSTE, mean_first_3min_FSTE_1wk; sem_first_3min_FSTI, sem_first_3min_FSTI_1wk; sem_first_3min_FSTC, sem_first_3min_FSTC_1wk]);
% Can use paired t-test because data points are paired across 48hr and 1wk sessions.
[h, p_3min_E, ci, stats] = ttest(mean(FSTE_tot(:,first_3min),2), mean(FSTE_1wk_tot(:,first_3min),2));
[h, p_3min_I, ci, stats] = ttest(mean(FSTI_tot(:,first_3min),2), mean(FSTI_1wk_tot(:,first_3min),2));
[h, p_3min_C, ci, stats] = ttest(mean(FSTC_tot(:,first_3min),2), mean(FSTC_1wk_tot(:,first_3min),2));
G = {[1 2], [4 5], [7 8]};
P = [p_3min_E, p_3min_I, p_3min_C];
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',14,'nosort',1,'max_errbar',max_sem/2));

set(gca,'XTick',[1:8]);
set(gca,'XTickLabels',{'48hr Exc', '1wk Exc', '', '48hr Inh', '1wk Inh', '', '48hr Ctl', '1wk Ctl'});
set(gca,'XTickLabelRotation',-45);

title('Test B first 3 minutes');

if saveit
	print(f, sprintf('%s\\%s.svg',file_path,file_str), '-dsvg');
	print(f, sprintf('%s\\%s.png',file_path,file_str), '-dpng');		
	disp(sprintf("done."));
end

%%%%%%%%%%%%
%% Test A %%
%%%%%%%%%%%% 

FSTI_TestA_mean = mean(mean(FSTI_TestA_tot,2));
FSTE_TestA_mean = mean(mean(FSTE_TestA_tot,2));
FSTC_TestA_mean = mean(mean(FSTC_TestA_tot,2));

FSTI_TestA_std = std(mean(FSTI_TestA_tot,2));
FSTE_TestA_std = std(mean(FSTE_TestA_tot,2));
FSTC_TestA_std = std(mean(FSTC_TestA_tot,2));

FSTI_TestA_sem = FSTI_TestA_std./sqrt(size(FSTI_TestA_tot,1));
FSTE_TestA_sem = FSTE_TestA_std./sqrt(size(FSTE_TestA_tot,1));
FSTC_TestA_sem = FSTC_TestA_std./sqrt(size(FSTC_TestA_tot,1));

FSTI_TestA_1wk_mean = mean(mean(FSTI_TestA_1wk_tot,2));
FSTE_TestA_1wk_mean = mean(mean(FSTE_TestA_1wk_tot,2));
FSTC_TestA_1wk_mean = mean(mean(FSTC_TestA_1wk_tot,2));

FSTI_TestA_1wk_std = std(mean(FSTI_TestA_1wk_tot,2));
FSTE_TestA_1wk_std = std(mean(FSTE_TestA_1wk_tot,2));
FSTC_TestA_1wk_std = std(mean(FSTC_TestA_1wk_tot,2));

FSTI_TestA_1wk_sem = FSTI_TestA_1wk_std./sqrt(size(FSTI_TestA_1wk_tot,1));
FSTE_TestA_1wk_sem = FSTE_TestA_1wk_std./sqrt(size(FSTE_TestA_1wk_tot,1));
FSTC_TestA_1wk_sem = FSTC_TestA_1wk_std./sqrt(size(FSTC_TestA_1wk_tot,1));

%
% Test A 

% The plot - paper
f=figure; hold on;
if for_paper
	f.PaperPosition = [0 0 2 2];
else
	f.PaperPosition = [0 0 4 4];
end
%set(gca,'FontSize',18);
file_str = '6a_testA_48hr_1wk_freezing';

b = bar([1 2 3], [FSTE_TestA_mean; FSTI_TestA_mean; FSTC_TestA_mean], 1.0); b.FaceColor='flat'; b.CData(1,:)=my_r; b.CData(2,:)=my_b, b.CData(3,:)=my_k; 
e = errorbar(1, FSTE_TestA_mean, FSTE_TestA_sem, 'Color', my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(2, FSTI_TestA_mean, FSTI_TestA_sem, 'Color', my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(3, FSTC_TestA_mean, FSTC_TestA_sem, 'Color', my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
max_sem = max([FSTE_TestA_sem, FSTI_TestA_sem, FSTC_TestA_sem]);
[p_EI, h_EI] = ranksum(mean(FSTE_TestA_tot,2), mean(FSTI_TestA_tot,2));
[p_EC, h_EC] = ranksum(mean(FSTE_TestA_tot,2), mean(FSTC_TestA_tot,2));
[p_IC, h_IC] = ranksum(mean(FSTI_TestA_tot,2), mean(FSTC_TestA_tot,2));

G = {[2 3], [1 2], [1 3]};
P = [p_IC, p_EI, p_EC];
%sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',14,'nosort',1,'max_errbar',max_sem));
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',8,'nosort',1,'max_errbar',max_sem));

%{
names = {'SST inhibition', 'SST activation', 'SST control   '};
data = {mean(FSTE_TestA_tot,2), mean(FSTI_TestA_tot,2), mean(FSTC_TestA_tot,2)};
stats_anova1(data, names, 'ANOVA');
stats_kruskalwallis(data, names, 'KW');
%}

b = bar([5 6 7], [FSTE_TestA_1wk_mean; FSTI_TestA_1wk_mean; FSTC_TestA_1wk_mean], 1.0); b.FaceColor='flat'; b.CData(1,:)=my_r; b.CData(2,:)=my_b, b.CData(3,:)=my_k; 
e = errorbar(5, FSTE_TestA_1wk_mean, FSTE_TestA_1wk_sem, 'Color', my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(6, FSTI_TestA_1wk_mean, FSTI_TestA_1wk_sem, 'Color', my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(7, FSTC_TestA_1wk_mean, FSTC_TestA_1wk_sem, 'Color', my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
max_sem = max([FSTE_TestA_1wk_sem, FSTI_TestA_1wk_sem, FSTC_TestA_1wk_sem]);
[p_EI, h_EI] = ranksum(mean(FSTE_TestA_1wk_tot,2), mean(FSTI_TestA_1wk_tot,2));
[p_EC, h_EC] = ranksum(mean(FSTE_TestA_1wk_tot,2), mean(FSTC_TestA_1wk_tot,2));
[p_IC, h_IC] = ranksum(mean(FSTI_TestA_1wk_tot,2), mean(FSTC_TestA_1wk_tot,2));

G = {[6 7], [5 6], [5 7]};
P = [p_IC, p_EI, p_EC];
%sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',14,'nosort',1,'max_errbar',max_sem));
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',8,'nosort',1,'max_errbar',max_sem));

%{
names = {'SST inhibition', 'SST activation', 'SST control   '};
data = {mean(FSTE_TestA_1wk_tot,2), mean(FSTI_TestA_1wk_tot,2), mean(FSTC_TestA_1wk_tot,2)};
stats_anova1(data, names, 'ANOVA');
stats_kruskalwallis(data, names, 'KW');
%}

set(gca,'XTick',[1:7]);
set(gca,'XTickLabels',{'Exc 48hr', 'Inh 48hr', 'Ctl 48hr', '', 'Exc 1wk', 'Inh 1wk', 'Ctl 1wk'});
set(gca,'XTickLabelRotation',-45);
ylim([0 60]);
ylabel('Freezing (%)');
title('Test A');

if saveit
	print(f, sprintf('%s\\%s.svg',file_path,file_str), '-dsvg');
	print(f, sprintf('%s\\%s.png',file_path,file_str), '-dpng');		
	disp(sprintf("done."));
end

%%
%% Test A 48hr vs 1wk
%%

f=figure; hold on;
if for_paper
	f.PaperPosition = [0 0 2 2];
else
	f.PaperPosition = [0 0 4 4];
end
%set(gca,'FontSize',18);
file_str = '6b_compare_48hr_1wk_TestA';

b = bar([1 2 4 5 7 8], [FSTE_TestA_mean; FSTE_TestA_1wk_mean; FSTI_TestA_mean; FSTI_TestA_1wk_mean; FSTC_TestA_mean; FSTC_TestA_1wk_mean], 1.0); 
%b.FaceColor='flat'; b.CData(1,:)=[1 0 0]; b.CData(2,:)=[1 0 0]; b.CData(3,:)=[0 0 1]; b.CData(4,:)=[0 0 1]; b.CData(5,:)=[0 0 0]; b.CData(6,:)=[0 0 0]; 
b.FaceColor='flat'; b.CData(1,:)=my_r; b.CData(2,:)=my_r; b.CData(3,:)=my_b; b.CData(4,:)=my_b; b.CData(5,:)=my_k; b.CData(6,:)=my_k; 
e = errorbar(1, FSTE_TestA_mean, FSTE_TestA_sem, 'Color', my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(2, FSTE_TestA_1wk_mean, FSTE_TestA_1wk_sem, 'Color', my_r); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(4, FSTI_TestA_mean, FSTI_TestA_sem, 'Color', my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(5, FSTI_TestA_1wk_mean, FSTI_TestA_1wk_sem, 'Color', my_b); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(7, FSTC_TestA_mean, FSTC_TestA_sem, 'Color', my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(8, FSTC_TestA_1wk_mean, FSTC_TestA_1wk_sem, 'Color', my_k); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;

%{
X=mean(FSTE_TestA_tot,2);
Y=mean(FSTE_TestA_1wk_tot,2);
scatter(ones(1,length(X)), X, 'k');
scatter(ones(1,length(Y))+1, Y, 'k');
for i=1:length(X)
	plot([1 2], [X(i), Y(i)], 'Color','k');
end

X=mean(FSTI_TestA_tot(1:5,:),2);
Y=mean(FSTI_TestA_1wk_tot,2);
scatter(ones(1,length(X))+3, X, 'k');
scatter(ones(1,length(Y))+4, Y, 'k');
for i=1:length(X)
	plot([4 5], [X(i), Y(i)], 'Color','k');
end

X=mean(FSTC_TestA_tot,2);
Y=mean(FSTC_TestA_1wk_tot,2);
scatter(ones(1,length(X))+6, X, [],my_h);
scatter(ones(1,length(Y))+7, Y, [], my_h);
for i=1:length(X)
	plot([7 8], [X(i), Y(i)], 'Color',my_h);
end
%}

max_sem = max([FSTE_TestA_sem, FSTE_TestA_1wk_sem, FSTI_TestA_sem, FSTI_TestA_1wk_sem, FSTC_TestA_sem, FSTC_TestA_1wk_sem]);
%{
[p_TestA_E, h_TestA_E] = ranksum(mean(FSTE_TestA_tot,2), mean(FSTE_TestA_1wk_tot,2));
[p_TestA_I, h_TestA_I] = ranksum(mean(FSTI_TestA_tot,2), mean(FSTI_TestA_1wk_tot,2));
[p_TestA_C, h_TestA_C] = ranksum(mean(FSTC_TestA_tot,2), mean(FSTC_TestA_1wk_tot,2));
%}
[h_TestA_E, p_TestA_E, ci, stats] = ttest(mean(FSTE_TestA_tot,2), mean(FSTE_TestA_1wk_tot,2));
[h_TestA_I, p_TestA_I, ci, stats] = ttest(mean(FSTI_TestA_tot,2), 	 mean(FSTI_TestA_1wk_tot,2));
[h_TestA_C, p_TestA_C, ci, stats] = ttest(mean(FSTC_TestA_tot,2), mean(FSTC_TestA_1wk_tot,2));

G = {[1 2], [4 5], [7 8]};
P = [p_TestA_E, p_TestA_I, p_TestA_C];
%sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',14,'nosort',1,'max_errbar',max_sem));
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',14,'nosort',1,'max_errbar',max_sem));
%

set(gca,'XTick',[1:8]);
set(gca,'XTickLabels',{'48hr Exc', '1wk Exc', '', '48hr Inh', '1wk Inh', '', '48hr Ctl', '1wk Ctl'});
set(gca,'XTickLabelRotation',-45);
ylim([0 50]);
ylabel('Freezing (%)');
title('Test A 48hr vs 1wk');

if saveit
	print(f, sprintf('%s\\%s.svg',file_path,file_str), '-dsvg');
	print(f, sprintf('%s\\%s.png',file_path,file_str), '-dpng');		
	disp(sprintf("done."));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Generalization - First 3min vs Test A, 48hr & 1wk
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f=figure; hold on;
if for_paper
	f.PaperPosition = [0 0 2 2];
else
	f.PaperPosition = [0 0 4 4];
end
%set(gca,'FontSize',18);
file_str = '7a_compare_First_3min_vs_TestA_48hr';

b = bar([1 2 4 5 7 8], [mean_first_3min_FSTE; FSTE_TestA_mean; mean_first_3min_FSTI; FSTI_TestA_mean; mean_first_3min_FSTC; FSTC_TestA_mean;], 1.0); 
%b.FaceColor='flat'; b.CData(1,:)=[1 0 0]; b.CData(2,:)=[1 0 0]; b.CData(3,:)=[0 0 1]; b.CData(4,:)=[0 0 1]; b.CData(5,:)=[0 0 0]; b.CData(6,:)=[0 0 0]; 
b.FaceColor='flat'; b.CData(1,:)=my_r; b.CData(2,:)=my_r; b.CData(3,:)=my_b; b.CData(4,:)=my_b; b.CData(5,:)=my_k; b.CData(6,:)=my_k; 
e = errorbar(1, mean_first_3min_FSTE, sem_first_3min_FSTE, 'r.'); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(2, FSTE_TestA_mean, FSTE_TestA_sem, 'r.'); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(4, mean_first_3min_FSTI, sem_first_3min_FSTI, 'b.'); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(5, FSTI_TestA_mean, FSTI_TestA_sem, 'b.'); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(7, mean_first_3min_FSTC, sem_first_3min_FSTE, 'k.'); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(8, FSTC_TestA_mean, FSTC_TestA_sem, 'k.'); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;

%{
X=mean(FSTE_tot(:,first_3min),2);
Y=mean(FSTE_TestA_tot,2);
scatter(ones(1,length(X)), X, 'k');
scatter(ones(1,length(Y))+1, Y, 'k');
for i=1:length(X)
	plot([1 2], [X(i), Y(i)], 'Color','k');
end

X=mean(FSTI_tot(:,first_3min),2);
Y=mean(FSTI_TestA_tot,2);
scatter(ones(1,length(X))+3, X, 'k');
scatter(ones(1,length(Y))+4, Y, 'k');
for i=1:length(X)
	plot([4 5], [X(i), Y(i)], 'Color','k');
end

X=mean(FSTC_tot(:,first_3min),2);
Y=mean(FSTC_TestA_tot,2);
scatter(ones(1,length(X))+6, X, [],my_h);
scatter(ones(1,length(Y))+7, Y, [], my_h);
for i=1:length(X)
	plot([7 8], [X(i), Y(i)], 'Color',my_h);
end
%}

max_sem = max([sem_first_3min_FSTE, FSTE_TestA_sem, sem_first_3min_FSTI, FSTI_TestA_sem, sem_first_3min_FSTE, FSTC_TestA_sem]);
%{
[p_TestA_E, h_TestA_E] = ranksum(mean(FSTE_TestA_tot,2), mean(FSTE_TestA_1wk_tot,2));
[p_TestA_I, h_TestA_I] = ranksum(mean(FSTI_TestA_tot,2), mean(FSTI_TestA_1wk_tot,2));
[p_TestA_C, h_TestA_C] = ranksum(mean(FSTC_TestA_tot,2), mean(FSTC_TestA_1wk_tot,2));
%}
[h_E, p_E, ci, stats] = ttest(mean(FSTE_tot(:,first_3min),2), mean(FSTE_TestA_tot,2));
[h_I, p_I, ci, stats] = ttest(mean(FSTI_tot(:,first_3min),2), mean(FSTI_TestA_tot,2));
[h_C, p_C, ci, stats] = ttest(mean(FSTC_tot(:,first_3min),2), mean(FSTC_TestA_tot,2));

G = {[1 2], [4 5], [7 8]};
P = [p_E, p_I, p_C];
%sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',14,'nosort',1,'max_errbar',max_sem));
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',8,'nosort',1,'max_errbar',max_sem));
%

set(gca,'XTick',[1:8]);
set(gca,'XTickLabels',{strcat('ExcFirst',first_min), 'Exc Test A', '', strcat('Inh First',first_min), 'Inh Test A', '', strcat('Ctl First',first_min), 'Ctl Test A'});
set(gca,'XTickLabelRotation',-45);
ylim([0 50]);
ylabel('Freezing (%)');
title(strcat('Test B First',first_min,'\newline vs Test A (48hr)'));

if saveit
	print(f, sprintf('%s\\%s.svg',file_path,file_str), '-dsvg');
	print(f, sprintf('%s\\%s.png',file_path,file_str), '-dpng');		
	disp(sprintf("done."));
end

mean_FSTC_tot_first_3min_per_mouse = ;
mean_FSTC_TestA_tot_per_mouse = ;

A = mean(FSTC_TestA_1wk_tot,2);
B = mean(FSTC_1wk_tot(:,first_3min),2);

DI = (A-B)./(A+B)
b = bar(DI);

figure; hold on;

% --- Boxplot ---
boxplot(DI, ...
    'Colors', 'k', ...
    'Widths', 0.4, ...
    'Symbol', '');   % remove default outlier symbols

% --- Jittered scatter ---
x_jitter = 1 + 0.08 * randn(size(DI));  % jitter around x = 1
scatter(x_jitter, DI, ...
    40, 'k', 'filled', ...
    'MarkerFaceAlpha', 0.7);

% --- Formatting ---
xlim([0.5 1.5])
xticks(1)
xticklabels({'DI (Novel vs Shocked)'})
ylabel('Discrimination Index')
yline(0, '--k', 'LineWidth', 1)

box off
set(gca, 'FontSize', 12)

mean_DI = mean(DI);
plot(1, mean_DI, 'rd', 'MarkerFaceColor', 'r', 'MarkerSize', 7)

% for purposes of matching with 1wk which didn't have it done for FSTE5
%FSTI_tot = [FSTI1; FSTI2([2],:); FSTI5; FSTI6(:,1:41)]; 
FSTI_tot = [FSTI1; FSTI2([2],:); FSTI6(:,1:41)];
FSTI_1wk_tot = [FSTI1_1wk; FSTI2_1wk([2],:); FSTI6_1wk];
% for purposes of matching with 1wk which didn't have it done for FSTE5
%FSTI_TestA_tot = [FSTI1_TestA; FSTI2_TestA([2],:); FSTI5_TestA; FSTI6_TestA];
FSTI_TestA_tot = [FSTI1_TestA; FSTI2_TestA([2],:); FSTI6_TestA];

A_48hr_C = mean(FSTC_TestA_tot,2); size(A_48hr_C,1)
B_48hr_C = mean(FSTC_tot(:,first_3min),2); size(B_48hr_C,1)
A_1wk_C = mean(FSTC_TestA_1wk_tot,2); size(A_1wk_C,1)
B_1wk_C = mean(FSTC_1wk_tot(:,first_3min),2); size(B_1wk_C,1)

A_48hr_I = mean(FSTI_TestA_tot,2); size(A_48hr_I,1)
B_48hr_I = mean(FSTI_tot(:,first_3min),2); size(B_48hr_I,1)
A_1wk_I = mean(FSTI_TestA_1wk_tot,2); size(A_1wk_I,1)
B_1wk_I = mean(FSTI_1wk_tot(:,first_3min),2); size(B_1wk_I,1)

A_48hr_E = mean(FSTE_TestA_tot,2); size(A_48hr_E,1)
B_48hr_E = mean(FSTE_tot(:,first_3min),2); size(B_48hr_E,1)
B_1wk_E = mean(FSTE_1wk_tot(:,first_3min),2); size(B_1wk_E,1)
A_1wk_E = mean(FSTE_TestA_1wk_tot,2); size(A_1wk_E,1)


% 1wk

f=figure; hold on;
if for_paper
	f.PaperPosition = [0 0 2 2];
else
	f.PaperPosition = [0 0 4 4];
end
%set(gca,'FontSize',18);
file_str = '7b_compare_First_3min_vs_TestA_1wk';

b = bar([1 2 4 5 7 8], [mean_first_3min_FSTE_1wk; FSTE_TestA_1wk_mean; mean_first_3min_FSTI_1wk; FSTI_TestA_1wk_mean; mean_first_3min_FSTC_1wk; FSTC_TestA_1wk_mean;], 1.0); 
%b.FaceColor='flat'; b.CData(1,:)=[1 0 0]; b.CData(2,:)=[1 0 0]; b.CData(3,:)=[0 0 1]; b.CData(4,:)=[0 0 1]; b.CData(5,:)=[0 0 0]; b.CData(6,:)=[0 0 0]; 
b.FaceColor='flat'; b.CData(1,:)=my_r; b.CData(2,:)=my_r; b.CData(3,:)=my_b; b.CData(4,:)=my_b; b.CData(5,:)=my_k; b.CData(6,:)=my_k; 
e = errorbar(1, mean_first_3min_FSTE_1wk, sem_first_3min_FSTE_1wk, 'r.'); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(2, FSTE_TestA_1wk_mean, FSTE_TestA_1wk_sem, 'r.'); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(4, mean_first_3min_FSTI_1wk, sem_first_3min_FSTI_1wk, 'b.'); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(5, FSTI_TestA_1wk_mean, FSTI_TestA_1wk_sem, 'b.'); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(7, mean_first_3min_FSTC_1wk, sem_first_3min_FSTE_1wk, 'k.'); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;
e = errorbar(8, FSTC_TestA_1wk_mean, FSTC_TestA_1wk_sem, 'k.'); e.CapSize=0; e.LineWidth=my_linewidth; %e.LineWidth=2;

%{
X=mean(FSTE_1wk_tot(:,first_3min),2);
Y=mean(FSTE_TestA_1wk_tot,2);
scatter(ones(1,length(X)), X, 'k');
scatter(ones(1,length(Y))+1, Y, 'k');
for i=1:length(X)
	plot([1 2], [X(i), Y(i)], 'Color','k');
end

X=mean(FSTI_1wk_tot(1:5,first_3min),2);
Y=mean(FSTI_TestA_1wk_tot,2);
scatter(ones(1,length(X))+3, X, 'k');
scatter(ones(1,length(Y))+4, Y, 'k');
for i=1:length(X)
	plot([4 5], [X(i), Y(i)], 'Color','k');
end

X=mean(FSTC_1wk_tot(:,first_3min),2);
Y=mean(FSTC_TestA_1wk_tot,2);
scatter(ones(1,length(X))+6, X, [],my_h);
scatter(ones(1,length(Y))+7, Y, [], my_h);
for i=1:length(X)
	plot([7 8], [X(i), Y(i)], 'Color',my_h);
end
%}

max_sem = max([sem_first_3min_FSTE_1wk, FSTE_TestA_1wk_sem, sem_first_3min_FSTI_1wk, FSTI_TestA_1wk_sem, sem_first_3min_FSTE_1wk, FSTC_TestA_1wk_sem]);
%{
[p_TestA_E, h_TestA_E] = ranksum(mean(FSTE_TestA_tot,2), mean(FSTE_TestA_1wk_tot,2));
[p_TestA_I, h_TestA_I] = ranksum(mean(FSTI_TestA_tot,2), mean(FSTI_TestA_1wk_tot,2));
[p_TestA_C, h_TestA_C] = ranksum(mean(FSTC_TestA_tot,2), mean(FSTC_TestA_1wk_tot,2));
%}
[p_E, h_E, stats] = signrank(mean(FSTE_1wk_tot(:,first_3min),2), mean(FSTE_TestA_1wk_tot,2));
[p_I, h_I, stats] = signrank(mean(FSTI_1wk_tot(1:5,first_3min),2), mean(FSTI_TestA_1wk_tot,2));
[p_C, h_C, stats] = signrank(mean(FSTC_1wk_tot(:,first_3min),2), mean(FSTC_TestA_1wk_tot,2));

G = {[1 2], [4 5], [7 8]};
P = [p_E, p_I, p_C];
%sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',14,'nosort',1,'max_errbar',max_sem));
sigstar('bar', G, P, struct('want_ticks',1,'sigbar_sep_amt',0.05,'FontSize',8,'nosort',1,'max_errbar',max_sem));
%

set(gca,'XTick',[1:8]);
set(gca,'XTickLabels',{strcat('Exc First',first_min), 'Exc Test A', '', strcat('Inh First',first_min), 'Inh Test A', '', strcat('Ctl First',first_min), 'Ctl Test A'});
set(gca,'XTickLabelRotation',-45);
ylim([0 80]);
ylabel('Freezing (%)');
title(strcat('Test B First',first_min,'\newlinevs Test A (1wk)'));

if saveit
	print(f, sprintf('%s\\%s.svg',file_path,file_str), '-dsvg');
	print(f, sprintf('%s\\%s.png',file_path,file_str), '-dpng');		
	disp(sprintf("done."));
end

%%
%% Some stats
%%

tone_FSTI_testB = mean(FSTI_tot(:,tone),2);
tone_FSTE_testB = mean(FSTE_tot(:,tone),2);
tone_FSTC_testB = mean(FSTC_tot(:,tone),2);
tone_FSTI_testB_1wk = mean(FSTI_1wk_tot(:,tone),2);
tone_FSTE_testB_1wk = mean(FSTE_1wk_tot(:,tone),2);
tone_FSTC_testB_1wk = mean(FSTC_1wk_tot(:,tone),2);

tone_FSTI_testB = mean(FSTI_tot(:,tone),2);
tone_FSTE_testB = mean(FSTE_tot(:,tone),2);
tone_FSTC_testB = mean(FSTC_tot(:,tone),2);
tone_FSTI_testB_1wk = mean(FSTI_1wk_tot(:,tone),2);
tone_FSTE_testB_1wk = mean(FSTE_1wk_tot(:,tone),2);
tone_FSTC_testB_1wk = mean(FSTC_1wk_tot(:,tone),2);

post_tone_FSTI_testB = mean(FSTI_tot(:,post_tone_only),2);
post_tone_FSTE_testB = mean(FSTE_tot(:,post_tone_only),2);
post_tone_FSTC_testB = mean(FSTC_tot(:,post_tone_only),2);
post_tone_FSTI_testB_1wk = mean(FSTI_1wk_tot(:,post_tone_only),2);
post_tone_FSTE_testB_1wk = mean(FSTE_1wk_tot(:,post_tone_only),2);
post_tone_FSTC_testB_1wk = mean(FSTC_1wk_tot(:,post_tone_only),2);

first_3min_FSTI_testB = mean(FSTI_tot(:,first_3min),2);
first_3min_FSTE_testB = mean(FSTE_tot(:,first_3min),2);
first_3min_FSTC_testB = mean(FSTC_tot(:,first_3min),2);
first_3min_FSTI_testB_1wk = mean(FSTI_1wk_tot(:,first_3min),2);
first_3min_FSTE_testB_1wk = mean(FSTE_1wk_tot(:,first_3min),2);
first_3min_FSTC_testB_1wk = mean(FSTC_1wk_tot(:,first_3min),2);

pval_tone = ranksum(tone_FSTI_testB, tone_FSTE_testB)
pval_post_tone = ranksum(post_tone_FSTI_testB, post_tone_FSTE_testB)
pval_tone_1wk = ranksum(tone_FSTI_testB_1wk, tone_FSTE_testB_1wk)
pval_post_tone_1wk = ranksum(post_tone_FSTI_testB_1wk, post_tone_FSTE_testB_1wk)
pval_first_3min = ranksum(first_3min_FSTI_testB, first_3min_FSTE_testB)
pval_first_3min_1wk = ranksum(first_3min_FSTI_testB_1wk, first_3min_FSTE_testB_1wk)

figure; hold on;
bar(1,mean_tone_FSTE,'r');
bar(2,mean_tone_FSTI,'b');
bar(3,mean_tone_FSTC,'k');
bar(4,mean_post_tone_only_FSTE,'r');
bar(5,mean_post_tone_only_FSTI,'b');
bar(6,mean_post_tone_only_FSTC,'k');

errorbar(1,mean_tone_FSTE,0,std_tone_FSTE,'k');
errorbar(2,mean_tone_FSTI,0,std_tone_FSTI,'k');
errorbar(3,mean_tone_FSTC,0,std_tone_FSTC,'k');
errorbar(4,mean_post_tone_only_FSTE,0,std_post_tone_only_FSTE,'k');
errorbar(5,mean_post_tone_only_FSTI,0,std_post_tone_only_FSTI,'k');
errorbar(6,mean_post_tone_only_FSTC,0,std_post_tone_only_FSTC,'k');

set(gca,'XTick',[1:6]);
set(gca,'XTickLabels',{'Tone','Tone','Tone','Post-tone','Post-tone','Post-tone'});
set(gca,'XTickLabelRotation',-45);
legend({'SST Exc','SST Inh','SST Ctl','','',''});
title('Test B (48 hours)');
ylabel('Freezing (%)');

figure; hold on;


bar(1,mean_tone_FSTE_1wk,'r');
bar(2,mean_tone_FSTI_1wk,'b');
bar(3,mean_tone_FSTC_1wk,'k');
bar(4,mean_post_tone_only_FSTE_1wk,'r');
bar(5,mean_post_tone_only_FSTI_1wk,'b');
bar(6,mean_post_tone_only_FSTC_1wk,'k');

errorbar(1,mean_tone_FSTE_1wk,0,std_tone_FSTE_1wk,'k');
errorbar(2,mean_tone_FSTI_1wk,0,std_tone_FSTI_1wk,'k');
errorbar(3,mean_tone_FSTC_1wk,0,std_tone_FSTC_1wk,'k');
errorbar(4,mean_post_tone_only_FSTE_1wk,0,std_post_tone_only_FSTE_1wk,'k');
errorbar(5,mean_post_tone_only_FSTI_1wk,0,std_post_tone_only_FSTI_1wk,'k');
errorbar(6,mean_post_tone_only_FSTC_1wk,0,std_post_tone_only_FSTC_1wk,'k');
set(gca,'XTick',[1:6]);
set(gca,'XTickLabels',{'Tone','Tone','Tone','Post-tone','Post-tone','Post-tone'});
set(gca,'XTickLabelRotation',-45);
legend({'SST Exc','SST Inh','SST Ctl','','',''});
title('Test B (1 week)');
ylabel('Freezing (%)');

%{
G = {[1,2], [3,4]};
P = [pval_tone, pval_post_tone];
sigstar('scatter', G, P, struct('want_ticks',1));
%}

names = {'SST inhibition', 'SST activation', 'SST control   '};

data = {tone_FSTI_testB, tone_FSTE_testB, tone_FSTC_testB};
stats_anova1(data, names, 'ANOVA Tones Test B');
stats_kruskalwallis(data, names, 'KW Tones Test B');

data = {post_tone_FSTI_testB, post_tone_FSTE_testB, post_tone_FSTC_testB};
stats_anova1(data, names, 'ANOVA Post-tone Test B');
stats_kruskalwallis(data, names, 'KW Post-tone Test B');

data = {tone_FSTI_testB_1wk, tone_FSTE_testB_1wk, tone_FSTC_testB_1wk};
stats_anova1(data, names, 'ANOVA Tones Test B 1wk');
stats_kruskalwallis(data, names, 'KW Tones Test B 1wk');

data = {post_tone_FSTI_testB_1wk, post_tone_FSTE_testB_1wk, post_tone_FSTC_testB_1wk};
stats_anova1(data, names, 'ANOVA Post-tone Test B 1wk');
stats_kruskalwallis(data, names, 'KW Post-tone Test B 1wk');



% Test A

FSTE1_TestA_t = [ 1.6 21.4 27.1]';
FSTE1_TestA_1wk_t = [ 0.2 21.9 30.8]';
FSTE2_TestA_t = [19.5 75.0]';
FSTE2_TestA_1wk_t = [8.1 18.9]';
FSTE3_TestA_t = [34.2 49.2]';
FSTE3_TestA_1wk_t = [18.1 21.0]';

FSTI1_TestA_t = [22.2 13.2 2.4 17.1]';
FSTI1_TestA_1wk_t = [9.4 6.4 1.5 13.3]';
FSTI2_TestA_t = [41.7 24.2]';
FSTI2_TestA_1wk_t = [52.2 10.7]';
FSTI4_TestA_t = [24.7 40.4 32.6 43.6]';
FSTI4_TestA_1wk_t = [7.6 16.7 7.6 26.1]';

FSTE_TestA_t = [FSTE1_TestA_t; FSTE2_TestA_t; FSTE3_TestA_t];
FSTE_TestA_1wk_t = [FSTE1_TestA_1wk_t; FSTE2_TestA_1wk_t; FSTE3_TestA_1wk_t];
FSTI_TestA_t = [FSTI1_TestA_t; FSTI2_TestA_t; FSTI4_TestA_t];
FSTI_TestA_1wk_t = [FSTI1_TestA_1wk_t; FSTI2_TestA_1wk_t; FSTI4_TestA_1wk_t];

dataE_t = {FSTE_TestA_t, FSTE_TestA_1wk_t};
namesE_t = {'hM3D Test A    ', 'hM3D Test A 1wk'};
stats_anova1(dataE_t', namesE_t', 'ANOVA hM3D Test A vs Test A 1wk');

dataI_t = {FSTI_TestA_t, FSTI_TestA_1wk_t};
namesI_t = {'hM4D Test A    ', 'hM4D Test A 1wk'};
stats_anova1(dataI_t, namesI_t, 'ANOVA hM4D Test A vs Test A 1wk');

function plot_scatter(X, Y, pos_X, pos_Y, c_X, c_Y)
	scatter(ones(1,length(X))+pos_X-1, X, 'Color',c_X);
	scatter(ones(1,length(Y))+pos_Y-1, Y, 'Color',c_Y);
	for i=1:length(X)
		plot([pos_X pos_Y], [X(i), Y(i)], 'Color',my_k);
	end
end

function plot_scatter3(X,Y,Z,pos_X,pos_Y,pos_Z,c_X,c_Y,c_Z)
	scatter(ones(1,length(X))+pos_X-1, X, 'Color',c_X);
	scatter(ones(1,length(Y))+pos_Y-1, Y, 'Color',c_Y);
	scatter(ones(1,length(Z))+pos_Z-1, Z, 'Color',c_Z);
	for i=1:length(X)
		plot([pos_X pos_Y], [X(i), Y(i)], 'k.');
		plot([pos_Y pos_Z], [Y(i), Z(i)], 'k.');
	end
end
