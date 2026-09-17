"""smolvla checks.

``overlap_matrix``      overlap support across eager/graph x plain/fused (4 rows)
``graph_flags``         every flag that reaches the graph path, crossed with graph on/off

``graph_flags`` exists for smolvla alone on purpose -- see ``tests/README.md`` (and the module
docstring) for why the other two backends do not need it.
"""
