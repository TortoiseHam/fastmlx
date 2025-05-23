Data Loader Notes:
* `mlx.data` can be used for GIL-free data pipelines. `Buffer.ordered_prefetch`
  spawns C++ threads and releases the GIL internally.
* `num_process=None` will use all CPU cores. A value of `0` disables
  parallelism.
Pipeline Ops Notes:
* Pipeline ops accept a state argument but the pipeline passes an empty dict since preprocessing typically has no global state.