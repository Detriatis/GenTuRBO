#!/usr/bin/env python3
"""Rebuild reported endpoint tables and symmetric leading-method comparisons."""
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import csv,json,math
from statistics import mean,stdev
from scipy.stats import t,ttest_rel,ttest_ind

def render(output_root):
    ROOT=Path(output_root)
    DATA=ROOT/'figures/data'
    LABELS={'tr_gp':'GenTuRBO','kermut':'GenTuRBO (Kermut)','tanimoto':'GenTuRBO (Tanimoto)','hamming_gp':'Hamming + GP/EI','hamming_gp_ei':'Hamming + GP/EI','turdbo':'GenTuRBO','turdbo_sequence':'GenTuRBO','adalead_official':'AdaLead','holo_ga':'Holo GA','adalead':'AdaLead','bo_evo':'BO-EVO','lambo2':'LaMBO-2 (online)','grid_ga':'Direct-grid GA'}
    EORDER=('tr_gp','hamming_gp','holo_ga','adalead_official','bo_evo','lambo2')
    PORDER=('kermut','tanimoto','hamming_gp','holo_ga','adalead','bo_evo','lambo2')
    SORDER=('turdbo_sequence','hamming_gp_ei','grid_ga','adalead_official')
    def read(name):
        with (DATA/name).open(newline='') as f:return list(csv.DictReader(f))
    def values(rows,keys):
        out=defaultdict(dict)
        for r in rows:
            key=tuple(r[k] for k in keys);seed=int(r['seed'])
            assert seed not in out[key]
            out[key][seed]=float(r['final_best'])
        for key,vs in out.items():assert set(vs)==set(range(20)),(key,len(vs))
        return {k:[v[s] for s in range(20)] for k,v in out.items()}
    E=values(read('ehrlich_sequence_baselines_n20_per_seed.csv'),['arm'])
    P=values(read('protein_comparison_per_seed.csv'),['task','method'])
    # Test every protein-method pair before looking up the observed leaders.
    # Holm's family includes all 4 * C(7, 2) = 84 possible protein contrasts.
    PAIRWISE=[]
    for task,display in [('gfp_medium','GFP-M'),('gfp_hard','GFP-H'),('aav_medium','AAV-M'),('aav_hard','AAV-H')]:
        for a_id,b_id in combinations(PORDER,2):
            a,b=P[(task,a_id)],P[(task,b_id)]
            result=ttest_ind(a,b,equal_var=False);ci=result.confidence_interval()
            PAIRWISE.append(dict(task=display,method_a=LABELS[a_id],method_b=LABELS[b_id],test='Welch',n_a=20,n_b=20,mean_a=mean(a),sample_sd_a=stdev(a),mean_b=mean(b),sample_sd_b=stdev(b),mean_difference=mean(a)-mean(b),t_statistic=float(result.statistic),df=float(result.df),p_two_sided=float(result.pvalue),ci95_low=float(ci.low),ci95_high=float(ci.high)))
    assert len(PAIRWISE)==84
    running=0.
    for rank,r in enumerate(sorted(PAIRWISE,key=lambda r:r['p_two_sided'])):
        running=max(running,min(1.,(len(PAIRWISE)-rank)*r['p_two_sided']))
        r['p_holm_all_84_protein_pairs']=running
    HOLM={(r['task'],frozenset((r['method_a'],r['method_b']))):r['p_holm_all_84_protein_pairs'] for r in PAIRWISE}
    with (DATA/'protein_all_pairwise_comparisons.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(PAIRWISE[0]),lineterminator='\n');w.writeheader();w.writerows(PAIRWISE)
    s=json.loads((ROOT/'figures/sokoban/figure_data.json').read_text())
    assert s['metric']=='best_box_switches'
    assert s['seeds']==list(range(20))
    S=s['endpoints']
    for vs in S.values():assert len(vs)==20
    TASKS=[('Ehrlich',{m:E[(m,)] for m in EORDER},EORDER,'paired',4)]
    for task,display in [('gfp_medium','GFP-M'),('gfp_hard','GFP-H'),('aav_medium','AAV-M'),('aav_hard','AAV-H')]:
        TASKS.append((display,{m:P[(task,m)] for m in PORDER if (task,m) in P},PORDER,'Welch',3))
    TASKS.append(('Sokoban',{m:S[m] for m in SORDER},SORDER,'paired',3))
    COMPARISONS=[];LEADERS={};STARS={}
    for task,vs,order,test,dp in TASKS:
        lead=max(vs,key=lambda m:mean(vs[m]));LEADERS[task]=lead
        group=[]
        for m in order:
            if m==lead or m not in vs:continue
            a,b=vs[lead],vs[m];delta=mean(a)-mean(b)
            if test=='paired':
                ds=[x-y for x,y in zip(a,b)];se=stdev(ds)/math.sqrt(20);df=19.;result=ttest_rel(a,b)
            else:
                va=stdev(a)**2/20;vb=stdev(b)**2/20;se=math.sqrt(va+vb);df=(va+vb)**2/(va**2/19+vb**2/19);result=ttest_ind(a,b,equal_var=False)
            radius=t.ppf(.975,df)*se
            r=dict(task=task,winner=LABELS[lead],baseline=LABELS[m],test=test,n_winner=20,n_baseline=20,winner_mean=mean(a),winner_sample_sd=stdev(a),baseline_mean=mean(b),baseline_sample_sd=stdev(b),mean_difference=delta,standard_error=se,t_statistic=float(result.statistic),df=df,p_two_sided=float(result.pvalue),ci95_low=delta-radius,ci95_high=delta+radius,positive_and_p_lt_005=delta>0 and result.pvalue<.05)
            group.append(r)
            r['p_holm_all_84_protein_pairs']=HOLM.get((task,frozenset((LABELS[lead],LABELS[m]))),'')
        STARS[task]=all(r['positive_and_p_lt_005'] for r in group)
        COMPARISONS.extend(group)
    assert len(COMPARISONS)==32
    with (DATA/'table2_all_external_comparisons.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(COMPARISONS[0]),lineterminator='\n');w.writeheader();w.writerows(COMPARISONS)
    def score(task,m,vs,dp):
        if m not in vs:return r'\textit{not run}'
        text=f'{mean(vs[m]):.{dp}f}\\pm{stdev(vs[m]):.{dp}f}'
        if m==LEADERS[task]:text=r'\mathbf{'+text+'}'
        if m==LEADERS[task] and STARS[task]:text+='^{*}'
        return '$'+text+'$'
    by_task={x[0]:x for x in TASKS}
    caption=r'''Bold marks the highest reported mean; $^*$ denotes a positive mean
    difference and $p<0.05$ against every other method in that task's table.
    Tests are exploratory and unadjusted (Supplementary Section~S6).'''
    # Protein table uses separate Medium and Hard task panels.
    parts=[r'''% Generated from all reported 20-seed endpoints by genturbo report all.
    \begin{table}[!htbp]
    \centering
    \footnotesize
    \setlength{\tabcolsep}{1.5pt}
    \renewcommand{\arraystretch}{1.05}
    \caption{\textbf{Protein fitness optimisation.} Best normalised CNN score
    after 2,000 charged evaluations, mean $\pm$ sample SD over 20 seeds.
    '''+caption+r''' Welch tests are used.}
    \label{tab:protein-headline}''']
    for label,a,b in [('Medium tasks','GFP-M','AAV-M'),('Hard tasks','GFP-H','AAV-H')]:
        if label=='Hard tasks':parts.append(r'\par\medskip')
        parts.append(r'\textbf{'+label+r'''}\par\smallskip
    \begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcc@{}}
    \toprule
    Method & GFP & AAV \\
    \midrule''')
        for m in PORDER:
            lab=LABELS[m]+(r'$^\dagger$' if m=='tanimoto' else '')
            parts.append(lab+' & '+score(a,m,by_task[a][1],3)+' & '+score(b,m,by_task[b][1],3)+r' \\')
        parts.append(r'''\bottomrule
    \end{tabular*}''')
    parts.append(r'''\par\smallskip\parbox{\columnwidth}{\footnotesize Hamming + GP/EI uses direct
    Hamming mutation proposals with the one-hot Tanimoto GP and marginal EI.
    $^\dagger$GFP-Medium GenTuRBO Tanimoto and Hamming + GP/EI fix noise variance to 0.001 in
    standardised outcome units; other reported Tanimoto tasks learn noise.
}
    \end{table}''')
    (ROOT/'figures/protein_baselines_table.tex').write_text('\n'.join(parts)+'\n')
    for task,file,title,endpoint,dp,extra in [
        ('Ehrlich','ehrlich_baselines_table.tex','Ehrlich sequence optimisation','Best reward',4,'GenTuRBO, Hamming and AdaLead use the spaced-pair GP. AdaLead selects by predicted mean.'),
        ('Sokoban','sokoban_baselines_table.tex','Sokoban coordination optimisation','Best minimum box-switch count',3,'GenTuRBO and Hamming share the best-parent, marginal-EI policy and full-history Tanimoto GP. AdaLead selects root-anchored rollouts by predicted mean. Direct-grid GA evaluates eight offspring per update. All operators can edit border cells.')]:
        _,vs,order,_,_=by_task[task]
        ls=[r'''% Generated from all reported 20-seed endpoints by genturbo report all.
    \begin{table}[H]
    \centering
    \footnotesize
    \setlength{\tabcolsep}{3pt}
    \renewcommand{\arraystretch}{1.05}''',r'\caption{\textbf{'+title+'.} '+endpoint+' after 2,000 charged\nevaluations, mean $\\pm$ sample SD over 20 seeds.\n'+caption+' Paired tests are used.}',r'\label{tab:'+('ehrlich' if task=='Ehrlich' else 'sokoban')+r'''-headline}
    \begin{tabular}{@{}lr@{}}
    \toprule''','Method & '+('Final best reward' if task=='Ehrlich' else 'Best $B$')+r' \\',r'\midrule']
        for m in order:ls.append(LABELS[m]+' & '+score(task,m,vs,dp)+r' \\')
        ls.extend([r'''\bottomrule
    \end{tabular}
    \par\smallskip\parbox{\columnwidth}{\footnotesize '''+extra+r'''}
    \end{table}'''])
        (ROOT/'figures'/file).write_text('\n'.join(ls)+'\n')
    def tex_p(p):
        if p>=.001:return f'{p:.4f}'
        exponent=math.floor(math.log10(p));return f'{p/10**exponent:.2f}'+r'\times10^{'+str(exponent)+'}'
    print(json.dumps({'comparisons':len(COMPARISONS),'leading_methods':{k:LABELS[v] for k,v in LEADERS.items()},'asterisks':STARS},indent=2))
