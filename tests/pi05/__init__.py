"""pi05 checks.

``overlap_matrix``      overlap support across eager/graph x plain/fused (4 rows)

The pi05 graph path packs the prefix differently from smolvla (the graph key is
``(slot_count, lang_len)``, so ``--pad-free`` and camera-slot packing change the key rather than
the captured body), which is why pi05 has no ``graph_flags`` check: see ``tests/README.md``.
"""
