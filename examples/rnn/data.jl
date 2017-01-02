using HDF5
using PyCall

type IRData
    X
    y
    test
    train_index
    val_index
    next_index
    n_epochs

    function IRData()
        s = new()
        s.X = map(Float32, h5read("data/ir_training_data.h5", "/features"))
        s.y = map(Float32, h5read("data/ir_training_data.h5", "/labels"))
        s.test = h5read("data/ir_testing_data.h5", "/features")

        s.train_index = randperm(length(s.y))
        s.n_epochs = 1
        s.next_index = 1
        s
    end
end

next_epoch!(data::IRData) = begin
    index = randperm(length(data.train_index))
    data.train_index = data.train_index[index]
    data.next_index = 1
    data.n_epochs += 1
    data
end

Base.getindex(data::IRData, i) = data.X[:, :, i], data.y[i]

next_batch(data::IRData, n) = begin
    i = data.next_index
    j = i + n -1 > length(data.train_index) ? length(data.train_index) : i + n -1
    index = data.train_index[i:j]
    data.next_index += n

    if j==length(data.train_index)
        next_epoch!(data)
    end
    data[index]
end