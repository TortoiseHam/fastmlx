Data Loader Notes:
* `mlx.data` can be used for GIL-free data pipelines. `Buffer.ordered_prefetch`
  spawns C++ threads and releases the GIL internally.
* `num_process=None` will use all CPU cores. A value of `0` disables
  parallelism.
Pipeline Ops Notes:
* Pipeline ops accept a state argument but the pipeline passes an empty dict since preprocessing typically has no global state.

Additional Notes:
* When the dataset already consists of MLX arrays (for example ``MLXDataset``),
  batching by direct array slicing avoids creating a ``mlx.data.Buffer``. This
  reduces memory usage and sidesteps some ``mlx.data`` event errors encountered
with very large datasets.

Training Optimization Notes:
* Using `mx.compile` to wrap the forward/backward/update step dramatically speeds up repeated calls.
* Capture `model.state`, `optimizer.state`, and `mx.random.state` as inputs/outputs so that weight updates persist between iterations.
Dataset Notes:
* CiFAIR datasets mirror CIFAR but with duplicates removed. Use simple urllib download from Google Drive when porting to FastMLX.
\nBug Fix Notes:\n* Google Drive downloads may serve an HTML confirmation page first. Use urllib to fetch the page, parse the confirm token, and retry the request before saving the file. Always validate that the resulting file is a zip archive using `zipfile.is_zipfile`.\n
