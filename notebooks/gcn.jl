### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# ╔═╡ d3c83370-2ca6-11f0-0011-9964ee40f932
import Pkg

# ╔═╡ 5d95170e-f869-410b-b734-99f1e05d51f6
Pkg.activate("./..")

# ╔═╡ 42dd5cce-f8b8-408f-b506-d690b0ee52b3
begin
    import QuantumGrav as QG
    import Flux
    import CUDA
    import GraphNeuralNetworks as GNN
    import CausalSets as CS
    import Arrow
    import Tables
    import StatsBase
    import Statistics
    import Random
    import ProgressMeter
    import MLUtils
    import CairoMakie
end

# ╔═╡ 43abff81-c8ab-4682-86bc-ed2793805121
md"Data loading and preprocessing"

# ╔═╡ 8c1689fe-41ba-4c2f-95b3-03d7bef94d98
path_to_data = joinpath(
    "/media", ENV["USER"], "dataLinux", "machinelearning_data", "QuantumGrav", "tiny")

# ╔═╡ 66355bea-3ced-42e5-a0e5-0d1650b2d34c
list_of_files = filter(x -> occursin(".arrow", x), readdir(path_to_data));

# ╔═╡ a6481fa9-e07b-4546-b420-ebca19cf1dfa
@info length(list_of_files)

# ╔═╡ a9cf0cfa-4394-446c-83a7-870086cb85e2
md"The fixed architecture used here is very far from optimal and only serves as a starting point and playing field. It has no skip connections and nothing, the normalizations are put in more or less randomly and training doesn't work either. Also, the decoder has dense layers only and hence has bad inductive bias. 

- optimize architecture and hyperparameters of this thing
- use more useful node features that actually carry information 
- use GATs and other kinds of layers that might be more suitable and produce better inductive bias
- add skip connections or other ways to allow the system to learn long-range connections 

- switch to different approaches: 
    - Hierarchical VAE 
    - d-VAE - specifically for DAGs. GraphNeuralNetworks.jl might not be suitable for this
    - Info/InfoMax VAE to avoid posterior collapse 
    - Riemann-VAE to turn latent-space into a non-blackbox system
	- find an architecture that has good inductive bias for partial orders
"

# ╔═╡ f14d7bb9-dcf8-413d-8259-29766c698cc1
begin
    mutable struct VAE{E, D}
        encoder::E
        latent::Flux.Parallel
        decoder::D
    end

    # fixed architecture for now
    function VAE(input_dim, latent_dim, output_dim; encoder_hidden_dims = [128, 80],
            decoder_hidden_dims = [80, 240, 680], activation = relu)
        # architecture is garbage, just here to make a pipeline work for now and 
        # learn how the packages can work together
        encoder = GNN.GNNChain(
            GNN.GCNConv(input_dim => encoder_hidden_dims[1], activation),
            Flux.LayerNorm(encoder_hidden_dims[1]),
            GNN.GCNConv(encoder_hidden_dims[1] => encoder_hidden_dims[2], activation),
            Flux.LayerNorm(encoder_hidden_dims[2]),
            GNN.GlobalPool(Statistics.mean)
        )

        latent = Flux.Parallel(
            tuple,
            # Mean, no activation to not restrict expressivity of mean 
            Flux.Dense(encoder_hidden_dims[2], latent_dim, identity),
            # Log variance, no activation to not restrict expressivity of log(sigma)
            Flux.Dense(encoder_hidden_dims[2], latent_dim, identity)
        )

        decoder = Flux.Chain(
            Flux.Dense(latent_dim, decoder_hidden_dims[1], activation),
            Flux.LayerNorm(decoder_hidden_dims[1]),
            Flux.Dense(decoder_hidden_dims[1], decoder_hidden_dims[2], activation),
            Flux.LayerNorm(decoder_hidden_dims[2]),
            Flux.Dense(decoder_hidden_dims[2], decoder_hidden_dims[3], activation),
            Flux.LayerNorm(decoder_hidden_dims[3]),
            Flux.Dense(decoder_hidden_dims[3], output_dim, Flux.sigmoid) # Output reconstruction
        )

        return VAE(encoder, latent, decoder)
    end

    Flux.@layer VAE

    # reparameterization trick to be able to differentiate through the sampling process
    function reparameterize(µ::AbstractArray, logσ²::AbstractArray)::AbstractArray
        # Sample from the standard normal distribution
        ε = CUDA::randn(size(µ))

        # Reparameterization trick
        z = µ .+ exp.(0.5f0 .* logσ²) .* ε

        return z
    end

    # Forward pass through the VAE
    function (vae::VAE)(g::GNN.GNNGraph, x::AbstractArray)
        # Encode input to get latent distribution parameters
        (µ, logσ²) = vae.encoder(g, x) |> vae.latent

        # Sample z from the latent distribution
        z = reparameterize(µ, logσ²)

        # Decode z to get reconstruction
        x̂ = vae.decoder(z)

        return x̂, µ, logσ²
    end
end

# ╔═╡ 53d1e462-4c37-418f-b7b7-9b001775dbb5
function encode_data(n_max, n_nodes, linkMatrix)::GNN.GNNGraph
    # we can construct other node features here if we need them. 

    # more meaningful node features: 

    # 1. Causal layer (approximate using in/out degree) -> we can use the size of the pre-and post relations for this
    # in_degree = vec(sum(linkMatrix, dims=1))
    # out_degree = vec(sum(linkMatrix, dims=2))

    # 2. what other features can we use? 

    return GNN.GNNGraph(
        linkMatrix; ndata = vcat(ones(Float32, n_nodes), zeros(Float32, n_max - n_nodes)))
end

# ╔═╡ 8a63e5f8-f47b-45bb-8b50-358212a8e2a2

# standard VAE loss function: reconstruction loss + KL divergence
function elbo_vae_loss(g::GNN.GNNGraph, model::VAE; beta::Float32 = 1.0f0)
    # Get the true adjacency matrix from the graph
    adj_true = GNN.adjacency_matrix(g) # This is the ground truth adjacency matrix

    # Flatten it to match decoder output format
    adj_true_flat = vec(adj_true)

    # Forward pass through model - x is just node features, not used in loss
    adj_pred_flat, μ, logσ² = model(g, reshape(g.ndata.x, (:, 1)))

    # Reconstruction loss
    rec_loss = Flux.logitbinarycrossentropy(adj_true_flat, adj_pred_flat, agg = sum) /
               size(g.ndata.x, 2)

    #  giving dim=2 here is not necessary anymore because we only have one dimension left
    kl_div = 0.5f0 .*
             Statistics.mean(sum(exp.(logσ²) .+ µ .* µ .- 1.0f0 .- logσ², dims = 1))
    return rec_loss + beta*kl_div, rec_loss, kl_div
end

# ╔═╡ b3347d24-5a11-4829-94f6-76405d32ba60

function elbo_vae_loss_batched(gs::AbstractArray, mode::VAE; beta::Float32 = 1.0f0)
    total = 0.0f0
    rec_loss = 0.0f0
    kl_div = 0.0f0
    for g in gs
        loss, rec, kl = elbo_vae_loss(g, mode; beta = beta)
        total += loss
        rec_loss += rec
        kl_div += kl
    end
    # mean loss over batch to normalize away batch size dependency
    return total/length(gs), rec_loss/length(gs), kl_div/length(gs)
end

# ╔═╡ 26553b75-9b12-404c-8011-a8ac37d2dd69
begin
    # training loop 

    # dataset
    utilized_data_proportion = 0.01
    num_files = Int(ceil(length(list_of_files) * utilized_data_proportion))

    @info "Using $(num_files) files from $(length(list_of_files))"

    dset = QG.DataLoader.Dataset(
        path_to_data,
        Random.shuffle(StatsBase.sample(list_of_files, num_files)),
        cache_size = 5,
        columns = [:linkMatrix, :n, :nmax]
    )
    n_max = Int(dset[1].nmax)

    # hyper parameters
    input_dim = n_max
    latent_dim = Int(ceil(n_max / 3))
    output_dim = input_dim*input_dim
    encoder_hidden_dims = [128, 80]
    decoder_hidden_dims = [80, 240, 680]
    activation = Flux.tanh # this is a critical choice
    batchsize = 128
    epochs = 1000
    beta = 1.0f0
    learning_rate = 1e-3
    # for early stopping
    current_loss = Inf
    last_loss = Inf
    patience = 15

    @info "input dim: $(input_dim), latent dim: $(latent_dim), output dim: $(output_dim)"
    @info "encoder hidden dims: $(encoder_hidden_dims), decoder hidden dims: $(decoder_hidden_dims)"

    # model
    device = CUDA.functional() ? Flux.gpu : Flux.cpu

    vae = VAE(input_dim, latent_dim, output_dim; encoder_hidden_dims = encoder_hidden_dims,
        decoder_hidden_dims = decoder_hidden_dims, activation = activation) |> device

    graphs = [encode_data(Int(dset[i].nmax), Int(dset[i].n), dset[i].linkMatrix)
              for i in 1:length(dset)]

    @info "num Graphs: $(length(graphs))"

    partition = MLUtils.splitobs(graphs, at = (0.7, 0.15, 0.149))[1:3]

    train_loader = Flux.DataLoader(
        partition[1],
        shuffle = true,
        batchsize = batchsize,
        collate = true,
        parallel = true
    )

    valid_loader = Flux.DataLoader(
        partition[2],
        shuffle = true,
        batchsize = batchsize,
        collate = true,
        parallel = true
    )

    test_loader = Flux.DataLoader(
        partition[3],
        shuffle = true,
        batchsize = batchsize,
        collate = true,
        parallel = true
    )

    @info "Training set size: $(length(train_loader)), Validation set size: $(length(valid_loader)), Test set size: $(length(test_loader))"

    # optimizer
    opt = Flux.setup(Flux.Adam(learning_rate), vae)

    v_loss_total = []
    train_loss_total = []
    v_loss_rec = []
    train_loss_rec = []
    v_loss_kl = []
    train_loss_kl = []

    # training itself
    for epoch in 1:epochs
        @info "Epoch: $epoch"

        all_loss_sum = 0.0f0
        rec_loss_sum = 0.0f0
        kl_loss_sum = 0.0f0

        ProgressMeter.@showprogress for (i, batch) in enumerate(train_loader)
            @debug "Batch: $(i)/$(length(train_loader))"

            batch_graphs = MLUtils.unbatch(batch) |> device

            (all_loss, rec_loss,
                kl_loss),
            (grads,) = Flux.withgradient(vae) do model
                elbo_vae_loss_batched(graphs, model; beta = beta)
            end

            # Check for numerical stability issues
            if any(isnan, all_loss)
                @warn "NaN detected in loss at epoch $epoch"
                hasnan = true
                break
            end
            all_loss_sum += all_loss
            rec_loss_sum += rec_loss
            kl_loss_sum += kl_loss

            # Update parameters
            Flux.update!(opt, vae, grads)
        end

        push!(train_loss_total, all_loss_sum/length(train_loader))

        push!(train_loss_rec, rec_loss_sum/length(train_loader))

        push!(train_loss_kl, kl_loss_sum/length(train_loader))

        v_l_total = 0.0f0
        v_l_rec = 0.0f0
        v_l_kl = 0.0f0

        for (i, batch) in enumerate(valid_loader)
            batch_graphs = MLUtils.unbatch(batch) |> device
            # Compute loss
            all_loss, rec_loss,
            kl_div = elbo_vae_loss_batched(batch_graphs, vae; beta = beta)

            # Check for numerical stability issues
            if any(isnan, all_loss)
                @warn "NaN detected in loss at epoch $epoch"
                hasnan = true
                break
            end

            v_l_total += all_loss
            v_l_rec += rec_loss
            v_l_kl += kl_div
        end

        push!(v_loss_total, v_l_total/length(valid_loader))
        push!(v_loss_rec, v_l_rec/length(valid_loader))
        push!(v_loss_kl, v_l_kl/length(valid_loader))

        @info "Epoch $epoch: Training Loss: $(train_loss_total[end]), Validation Loss: $(v_loss_total[end]), patience: $(patience)"

        global current_loss = v_loss_total[end]

        if current_loss < last_loss*0.99
            global patience = 15
            global last_loss = current_loss
        else
            global patience -= 1
            if patience == 0
                @info "Early stopping at epoch $epoch"
                break
            end
        end
    end
end

# ╔═╡ db2e6576-9373-4503-ad8f-2e7ba860e4a7
begin
    fig = CairoMakie.Figure(size = (800, 600))
    ax1 = CairoMakie.Axis(fig[1, 1], xlabel = "Epochs", ylabel = "Training loss total")
    ax2 = CairoMakie.Axis(fig[1, 2], xlabel = "Epochs", ylabel = "Training loss rec")
    ax3 = CairoMakie.Axis(fig[1, 3], xlabel = "Epochs", ylabel = "Training loss kl")
    axtop = [ax1, ax2, ax3]

    ax4 = CairoMakie.Axis(fig[2, 1], xlabel = "Epochs", ylabel = "Validation loss total")
    ax5 = CairoMakie.Axis(fig[2, 2], xlabel = "Epochs", ylabel = "Validation loss rec")
    ax6 = CairoMakie.Axis(fig[2, 3], xlabel = "Epochs", ylabel = "Validation loss kl")
    axbottom = [ax4, ax5, ax6]
    ax = [axtop, axbottom]
    ax = hcat(ax...)
end

# ╔═╡ 1c9c7e31-c300-42b2-9c0c-d8a89b957db6
# color = CairoMakie.wong_colors()
begin
    lines!(ax[1, 1], 1:length(train_loss_total), train_loss_total, label = "Train Loss")
    lines!(ax[2, 1], 1:length(train_loss_rec), train_loss_rec, label = "Train Rec Loss")
    lines!(ax[3, 1], 1:length(train_loss_kl), train_loss_kl, label = "Train KL Loss")

    lines!(ax[1, 2], 1:length(v_loss_total), v_loss_total, label = "Train Loss")
    lines!(ax[2, 2], 1:length(v_loss_rec), v_loss_rec, label = "Train Rec Loss")
    lines!(ax[3, 2], 1:length(v_loss_kl), v_loss_kl, label = "Train KL Loss")

    fig
end

# ╔═╡ 035848db-77c4-4bb6-8fe7-dcc970d599b0
md"validation loss is unstable, so training doesn't really work here yet. Architecture is not good."

# ╔═╡ Cell order:
# ╠═d3c83370-2ca6-11f0-0011-9964ee40f932
# ╠═5d95170e-f869-410b-b734-99f1e05d51f6
# ╠═42dd5cce-f8b8-408f-b506-d690b0ee52b3
# ╟─43abff81-c8ab-4682-86bc-ed2793805121
# ╠═8c1689fe-41ba-4c2f-95b3-03d7bef94d98
# ╠═66355bea-3ced-42e5-a0e5-0d1650b2d34c
# ╠═a6481fa9-e07b-4546-b420-ebca19cf1dfa
# ╟─a9cf0cfa-4394-446c-83a7-870086cb85e2
# ╠═f14d7bb9-dcf8-413d-8259-29766c698cc1
# ╠═53d1e462-4c37-418f-b7b7-9b001775dbb5
# ╠═8a63e5f8-f47b-45bb-8b50-358212a8e2a2
# ╠═b3347d24-5a11-4829-94f6-76405d32ba60
# ╠═26553b75-9b12-404c-8011-a8ac37d2dd69
# ╠═db2e6576-9373-4503-ad8f-2e7ba860e4a7
# ╠═1c9c7e31-c300-42b2-9c0c-d8a89b957db6
# ╠═035848db-77c4-4bb6-8fe7-dcc970d599b0
