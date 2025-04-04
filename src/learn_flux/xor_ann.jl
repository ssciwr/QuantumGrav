using Flux, Statistics, ProgressMeter, CUDA, cuDNN
device = gpu_device() # get the GPU device
println("gpu device: ", device, " cuda functional: ", CUDA.functional())

# generate some data for xor problem 
noisy = rand(Float32, 2, 1000) # 2x1000 matrix 
truth = [xor(col[1] > 0.5, col[2] > 0.5) for col in eachcol(noisy)] # 1000x1 vector{bool}

# UNDERSTAND WHAT HAS BEEN MARKED AS TODO HER!!

# define model-> one hidden layer multilayer perceptron
# TODO: need to understand this better, forgot most of it ... 
model = Chain(
    Dense(2 => 3, tanh), # 2 inputs, 3 hidden units. the activation function is integrated here # TODO: why does tanh work better here than relu? 
    BatchNorm(3), # batch normalization. Check again what this realy does, I have forgotten
    Dense(3 => 2)
) |> device # move model to GPU 

# model encapsulates the parameters and initializes them randomly
gpu_data = noisy |> device
out1 = model(gpu_data)  # move data to GPU
probs1 = softmax(out1) |> cpu# apply softmax to get probabilities 

# we use one-hot encoding for the and batches of 64 samples for training 
target = Flux.onehotbatch(truth, [true, false]) # TODO: understand this better again. 
loader = Flux.DataLoader((noisy, target), batchsize = 64, shuffle = true)

# use an adam optimizer and store the momentum etc 
# TODO: this is relatively opaque, what does a call to this thing entail?
opt_state = Flux.setup(Flux.Adam(0.01), model) # 0.01 is the learning rate

# training loop 
losses = []
@showprogress for epoch in 1:1_000
    for xy_cpu in loader
        # unpack batch of data, move to gpu 
        x, y = xy_cpu |> device
        loss, gradients = Flux.withgradient(model) do m
            y_hat = m(x)
            Flux.logitcrossentropy(y_hat, y)
        end
        Flux.update!(opt_state, model, gradients[1])
        push!(losses, loss)  # logging, outside gradient context
    end
end

out2 = model(noisy |> device) # move data to GPU
probs2 = softmax(out2) |> cpu # apply softmax to get probabilities
accuracy = mean((probs2[1, :] .> 0.5) .== truth) # compute accuracy

println("accuracy: ", accuracy)

# plot some shit
using AlgebraOfGraphics, CairoMakie, DataFrames

set_theme!(theme_minimal())

# Create loss plot over iteration steps
# Create separate dataframes
batch_df = DataFrame(
    x = 1:length(losses),
    y = losses,
    type = fill("Per Batch", length(losses))
)

epoch_df = DataFrame(
    x = epochmeandf.epoch,
    y = epochmeandf.mean_over_epoch,
    type = fill("Epoch Mean", length(epochmeandf.epoch))
)

# Create separate plots and combine them with + operator
# This ensures the epoch mean (added second) is drawn on top
batch_plot = data(batch_df) * mapping(:x, :y, color = :type) *
             visual(Lines, linewidth = 1, alpha = 0.5)
epoch_plot = data(epoch_df) * mapping(:x, :y, color = :type) * visual(Lines, linewidth = 1)

# Combine plots (+ controls the order)
combined_plot = batch_plot + epoch_plot

# Draw with a legend
draw(combined_plot, axis = (; xscale = log10, xlabel = "batch", ylabel = "loss"))

# create before-after plot for results

truth_df = DataFrame(
    x = noisy[1, :],
    y = noisy[2, :],
    type = truth
)

probs_before_training = DataFrame(
    x = noisy[1, :],
    y = noisy[2, :],
    type = probs1[1, :]
)

probs_after_training = DataFrame(
    x = noisy[1, :],
    y = noisy[2, :],
    type = probs2[1, :]
)

truth_plot = data(truth_df) * mapping(:x, :y, color = :type) * visual(Scatter)
draw(truth_plot, axis = (; xlabel = "x", ylabel = "y", title = "Truth"))

before_plot = data(probs_before_training) * mapping(:x, :y, color = :type) * visual(Scatter)
draw(before_plot, axis = (; xlabel = "x", ylabel = "y", title = "Before Training"))

after_plot = data(probs_after_training) * mapping(:x, :y, color = :type) * visual(Scatter)
draw(after_plot, axis = (; xlabel = "x", ylabel = "y", title = "After Training"))
