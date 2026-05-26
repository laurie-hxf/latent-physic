Legacy Newton entry points.

These scripts are older standalone demos or parameter-fitting experiments that are
not imported by the current MuJoCo contact-friction training, replay, or
visualization pipeline.

- `PBD.py`: segmented point-cloud PBD rollout entry point.
- `fit_demo_params.py`: ManiSkill demo parameter fitting experiment.
- `newton_box_demo.py`: standalone Newton box-force demo.

Keep shared helper modules such as `pbd_math.py`, `pbd_types.py`, and
`newton_surface_points_demo.py` at the top level because active friction fitting
code still imports them.
