# app_callbacks_improved.py — Callbacks pour les nouvelles visualisations

from dash import Input, Output
from app_core_improved import (
    app,
    create_competency_radar_by_semester,
    create_competency_flow_sankey,
    create_workload_heatmap,
    create_competency_progression_timeline,
    create_competency_module_matrix,
    create_competency_sunburst,
    create_competency_stats_cards
)

@app.callback(
    Output('stats-cards-container', 'children'),
    Input('stats-cards-container', 'id')  # Dummy input pour le chargement initial
)
def update_stats_cards(_):
    return create_competency_stats_cards()

@app.callback(
    Output('progression-timeline', 'figure'),
    Input('progression-timeline', 'id')
)
def update_progression(_):
    return create_competency_progression_timeline()

@app.callback(
    Output('radar-chart', 'figure'),
    Input('radar-chart', 'id')
)
def update_radar(_):
    return create_competency_radar_by_semester()

@app.callback(
    Output('workload-heatmap', 'figure'),
    Input('workload-heatmap', 'id')
)
def update_workload(_):
    return create_workload_heatmap()

@app.callback(
    Output('sankey-diagram', 'figure'),
    Input('sankey-diagram', 'id')
)
def update_sankey(_):
    return create_competency_flow_sankey()

@app.callback(
    Output('sunburst-chart', 'figure'),
    Input('sunburst-chart', 'id')
)
def update_sunburst(_):
    return create_competency_sunburst()

@app.callback(
    Output('matrix-chart', 'figure'),
    Input('matrix-chart', 'id')
)
def update_matrix(_):
    return create_competency_module_matrix()

if __name__ == '__main__':
    app.run_server(debug=True)