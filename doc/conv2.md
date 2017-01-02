## Convolutional Neural Networks
Convolutions exist in seep an an extension of `Base.conv2` (although arbitrary
dimensions are supported).  The channel (also known as filter) must be the
first index of the data array.  The filter array must be (channel_out &times;
channel_in &times; dim2...).  The output will have the same number of
dimensions af the input.

```julia
x = make_node(1, 28, 28, 10) # Define input with 1 channel, 28 rows, 28 columns, and 10 images
const w = make_node(randn(8, 1, 5, 5)/sqrt(25)) # 8 outputs, 1 channel 5 row × 5 column input
const b = make_node(zeros(8)) # 8 bias elements
y = conv2(w, x) .+ b # Apply the convolution, output is 8 channels × 23 rows × 23 columns × 10 images
a = Seep.max!(0, y) # Seep.max!(0, ...) is ReLU
p = Seep.pool(a, (1,2,2)) # pool size is 1 channel × 2 rows × 2 columns (default stride is the pool size)
```
