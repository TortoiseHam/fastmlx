FastMLX is a new project attempting to port the syntax of FastEstimator onto the MLX backend.

FastEstimator is a fancier version of Keras which has TensorFlow and PyTorch backends. The FastEstimator codebase is complicated and not well represented in your training data, so read through it carefully to understand how everything works before attempting to port features or cut corners. Because FastEstimator uses TF and Torch, it distinguishes between Numpy operations (used in data pipeline) and Tensor operations (used in model training). When porting to FastMLX, there is no longer any difference between these two concepts as everything should just be an MLX array. 

Be careful to always use rigorous type hinting. Look for high quality solutions rather than shortcuts.

If you learn anything useful about the FastEstimator, MLX, or FastMLX codebases which is worth recording for future reference (you will be doing additional code-porting tasks in the future), you can use the rest of this document below the line as a scratchpad in order to store persistent tips for your future self. 

---

