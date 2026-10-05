"""Rebuild the Sokoban figures from their saved 20-seed source data."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from genturbo.figures.style import (COLOURS, LINE_STYLES, PAGE_WIDTH, apply_style,
                          style_axis, legend_above, save_figure)


def render(output_root):
    root=Path(output_root)
    figure_dir=root/'figures'
    data=json.loads((figure_dir/'sokoban/figure_data.json').read_text())
    winner=json.loads((figure_dir/'sokoban/winner.json').read_text())
    generator_winner=json.loads((figure_dir/'sokoban/generator_winner.json').read_text())
    assert generator_winner['objective_status']=='exact'
    query_grid=np.asarray(data['queries'])
    curves={method:np.asarray(values) for method,values in data['curves'].items()}
    endpoints=data['endpoints']
    methods=['turdbo_sequence','hamming_gp_ei','generator','grid_ga','adalead_official']
    labels=['GenTuRBO','Hamming + GP/EI','Independent sampling','Direct-grid GA','AdaLead']
    colors=[COLOURS[method] for method in methods]
    assert data['metric']=='best_box_switches'
    assert len(data['seeds']) >= 2
    for method in methods:
        assert curves[method].shape==(len(data['seeds']),len(query_grid))
        assert np.array_equal(curves[method][:,-1],endpoints[method])
    COLORS = {"wall": "#26394B", "floor": "#F3EEE3", "goal": "#9658B5",
              "box": "#DFA147", "box_edge": "#8D591F", "player": "#167D98"}


    def draw_tile(ax, x, y, tile):
        from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

        wall = tile == 0
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor=COLORS["wall" if wall else "floor"],
                               edgecolor="#425263" if wall else "#DDD6C8", linewidth=.45))
        if wall:
            ax.plot([x + .12, x + .88], [y + .81, y + .81], color="#344A5D", lw=.65)
            return
        if tile in (4, 5, 6):
            ax.add_patch(Circle((x + .5, y + .5), .30, facecolor="#E9D8F0",
                                edgecolor=COLORS["goal"], linewidth=1.7))
            ax.add_patch(Circle((x + .5, y + .5), .095, facecolor=COLORS["goal"], edgecolor="none"))
        if tile in (3, 5):
            ax.add_patch(FancyBboxPatch((x + .14, y + .14), .72, .72,
                                        boxstyle="round,pad=0,rounding_size=.06",
                                        facecolor=COLORS["box"],
                                        edgecolor=COLORS["goal"] if tile == 5 else COLORS["box_edge"],
                                        linewidth=2 if tile == 5 else 1.2))
            ax.plot([x + .25, x + .75], [y + .25, y + .75], color=COLORS["box_edge"], lw=1)
            ax.plot([x + .25, x + .75], [y + .75, y + .25], color=COLORS["box_edge"], lw=1)
        if tile in (2, 6):
            ax.add_patch(Circle((x + .5, y + .5), .34, facecolor=COLORS["player"],
                                edgecolor="white", linewidth=1.2))
            ax.add_patch(Circle((x + .5, y + .39), .10, facecolor="white", edgecolor="none"))
            ax.add_patch(FancyBboxPatch((x + .36, y + .52), .28, .14,
                                        boxstyle="round,pad=0,rounding_size=.06",
                                        facecolor="white", edgecolor="none"))


    apply_style()
    def board_axis(ax, puzzle):
        for i,tile in enumerate(puzzle['canonical_board']):
            draw_tile(ax,i%10,i//10,tile)
        ax.set(xlim=(0,10),ylim=(10,0),aspect='equal');ax.axis('off')
    fig,axes=plt.subplots(1,3,figsize=(PAGE_WIDTH,2.8),gridspec_kw={'width_ratios':[1.2,1,1]},layout='constrained')
    for method,label,color in zip(methods,labels,colors,strict=True):
        values=curves[method];mean=values.mean(0);sd=values.std(0,ddof=1)
        axes[0].step(query_grid,mean,where='post',label=label,color=color,
                     lw=1.7 if method=='turdbo_sequence' else 1.15,ls=LINE_STYLES[method],
                     zorder=4 if method=='turdbo_sequence' else 3)
        axes[0].fill_between(query_grid,mean-sd,mean+sd,step='post',color=color,
                             alpha=.14 if method=='turdbo_sequence' else .08,lw=0)
    axes[0].axvline(64,color='#aaaaaa',ls=':',lw=.8)
    axes[0].set(xlim=(0,2000),ylim=(0,16),xticks=[0,1000,2000],xlabel='Charged evaluations',ylabel='Best switches so far, B',title='(a) Search progress')
    style_axis(axes[0])
    board_axis(axes[1],generator_winner);axes[1].set_title(f"(b) Best RS: B = {generator_winner['score']}",pad=8)
    board_axis(axes[2],winner);axes[2].set_title(f"(c) Best GenTuRBO: B = {winner['score']}",pad=8)
    legend_above(fig, [axes[0]], ncol=3)
    save_figure(fig, figure_dir/'sokoban_comparison', formats=('pdf','svg','png'))
    svg_path = figure_dir / 'sokoban_comparison.svg'
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
