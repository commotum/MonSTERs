Eventually my paper will be something that covers the following:

1. RoPE was a breakthrough (why? it allowed a modification-light or drop-in ready adaptation for LLMs positional encoders creating relative absolute fusion for linear attention methods.) 

2. Since then it has evoked many variants in an effort to port the benefits of the transformer to other domains including vision. This includes axial RoPE. However, no singular adaptation has gained dominance over others, and many efforts simply do row-major indexing even in 2D task domains.

3. We show that the transformer/attention architecture is so powerful, that when building models restricted to a pre-determined "context-length" using a single type of data, the encoding method appears to make little to no difference on performance for most tasks. The important component of the positional encoder is that the model is provided some type of discernable texture unique to each position that remains constant throughout training.

4. This texture however requires a highly human dependent step, translating a task's domain into tokens and positional indices so that it fits in with models that currently exist. If you want to change the task domain at all, it requires retraining.

5. additionally, even axial RoPE fails to work on diagonals, since the two dimensions are independent.

6. MonSTERs is the solution