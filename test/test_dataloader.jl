using TestItems

@testsnippet MakeData begin
    import QuantumGrav
    import Arrow

    function makeMockData(nf, nd)
        # Create a temporary directory for testing
        temp_dir = mktempdir()

        for i in 1:nf
            # Create a sample Arrow file with dummy data
            file_path = joinpath(temp_dir, "test_data_$(i).arrow")
            data = [rand(Float32, 10, 10) for _ in 1:nd]
            other_data = [rand(Float32) for _ in 1:nd]
            Arrow.write(file_path, (linkMatrix = data, otherColumn = other_data))
        end
        return temp_dir
    end

    dir = makeMockData(3, 50)
end

@testitem "DataLoader.constructQuantumGravDataset" tags=[:dataloader] setup=[MakeData] begin

    # Initialize the QuantumGravDataset
    dataset=QuantumGrav.DataLoader.QuantumGravDataset(
        dir, collect(readdir(dir)); batch_size = 5, cache_size = 2)

    # Test the constructor
    @test dataset.base_path == dir
    @test dataset.file_paths ==
          ["test_data_1.arrow", "test_data_2.arrow", "test_data_3.arrow"]
    @test dataset.batch_size == 5
    @test dataset.file_length == 50
    @test length(dataset.indices) == 3 * 50
    @test dataset.max_buffer_size == 2
end

@testitem "DataLoader.loadData" tags=[:dataloader] setup=[MakeData] begin
    import Tables
    dataset=QuantumGrav.DataLoader.QuantumGravDataset(
        dir, collect(readdir(dir)); batch_size = 5, cache_size = 2)

    # Test loading data from the first file
    data=QuantumGrav.DataLoader.loadData(dataset, 1)
    @test length(data) == 50
    @test length(data[1]) == 2
    @test collect(keys(data[1])) == [:linkMatrix, :otherColumn]
    @test keys(data) == 1:50
end

@testitem "DataLoader.getindex" tags=[:dataloader] setup=[MakeData] begin
    dataset=QuantumGrav.DataLoader.QuantumGravDataset(
        dir, collect(readdir(dir)); batch_size = 5, cache_size = 2)

    # Test getting an index from the dataset
    first=dataset[1]
    @test collect(keys(first)) == [:linkMatrix, :otherColumn]
    @test length(dataset.buffer) == 1

    # Test getting a batch of data at once
    multiple=dataset[[1, 45, 75, 142]]
    @test length(multiple) == 4
    @test length(multiple[1].linkMatrix) == 100
    @test length(multiple[1].otherColumn) == 1
    @test length(dataset.buffer) == 2
end

@testitem "DataLoader.collateMatrices" tags=[:dataloader] setup=[MakeData] begin

    # Test collating matrices
    rows=[8, 9, 12, 10, 7]

    columns=[12, 9, 11, 12, 10]

    data=[rand(Float32, rows[i], columns[i]) for i in 1:5]

    collated_data=QuantumGrav.DataLoader.collateMatrices(data)

    @test size(collated_data) == (12, 12, 5)

    # Test collating matrices with equal sizes --> only put together 
    data=[rand(Float32, 10, 10) for i in 1:5]

    collated_data=QuantumGrav.DataLoader.collateMatrices(data)

    @test size(collated_data) == (10, 10, 5)
end
