using TestItems

@testsnippet MakeData begin
    import QuantumGrav
    import Arrow
    import JLD2

    function makeMockData(nf, nd)
        # Create a temporary directory for testing
        temp_dir = mktempdir()

        for i = 1:nf
            # Create a sample Arrow file with dummy data
            file_path = joinpath(temp_dir, "test_data_$(i).arrow")
            data = [rand(Float32, 10, 10) for _ = 1:nd]
            other_data = [rand(Float32) for _ = 1:nd]
            Arrow.write(file_path, (link_matrix = data, otherColumn = other_data))
        end

        file_path = joinpath(temp_dir, "test_data.jld2")
        JLD2.jldopen(file_path, "w") do file

            for i = 1:nf
                link_matrix = [rand(Float32, 10, 10) for _ = 1:nd]
                otherColumn = [rand(Float32) for _ = 1:nd]
                file["chunk$(i)/link_matrix"] = link_matrix
                file["chunk$(i)/otherColumn"] = otherColumn
            end
        end
        return temp_dir
    end

    dir = makeMockData(3, 50)
end

@testitem "DataLoader.constructDataset_arrow" tags=[:dataloader] setup=[MakeData] begin
    # Initialize the Dataset
    Dataset=QuantumGrav.DataLoader.Dataset(dir; cache_size = 2, mode = "arrow")

    # Test the constructor
    @test Dataset.base_path == dir
    @test Dataset.file_paths ==
          ["test_data_1.arrow", "test_data_2.arrow", "test_data_3.arrow"]
    @test Dataset.file_length == 50
    @test length(Dataset.indices) == 3 * 50
    @test Dataset.max_buffer_size == 2
    @test Dataset.mode == "arrow"
end

@testitem "DataLoader.constructDataset_jld2" tags=[:dataloader] setup=[MakeData] begin

    # Initialize the Dataset
    Dataset=QuantumGrav.DataLoader.Dataset(dir; cache_size = 2, mode = "jld2")

    # Test the constructor
    @test Dataset.base_path == dir
    @test Dataset.file_paths == "test_data.jld2"
    @test Dataset.file_length == 50
    @test length(Dataset.indices) == 3 * 50
    @test Dataset.max_buffer_size == 2
    @test Dataset.mode == "jld2"
end

@testitem "DataLoader.load_data arrow" tags=[:dataloader] setup=[MakeData] begin
    import Tables
    Dataset=QuantumGrav.DataLoader.Dataset(dir; cache_size = 2, mode = "arrow")

    # Test loading data from the first file
    data=QuantumGrav.DataLoader.load_data(Dataset, 1)
    @test length(data) == 2 # 2 columns
    @test length(data.link_matrix) == 50
    @test length(data.otherColumn) == 50
    @test collect(keys(data)) == [:link_matrix, :otherColumn]
end

@testitem "DataLoader.load_data jld2" tags=[:dataloader] setup=[MakeData] begin
    import Tables
    Dataset=QuantumGrav.DataLoader.Dataset(dir; cache_size = 2, mode = "arrow")

    # Test loading data from the first file
    data=QuantumGrav.DataLoader.load_data(Dataset, 1)
    @test length(data) == 2
    @test length(data.link_matrix) == 50
    @test length(data.otherColumn) == 50
    @test collect(keys(data)) == [:link_matrix, :otherColumn]
end

@testitem "DataLoader.getindex arrow" tags=[:dataloader] setup=[MakeData] begin
    Dataset=QuantumGrav.DataLoader.Dataset(dir; cache_size = 2)

    # Test getting an index from the Dataset
    first=Dataset[1]
    @test collect(keys(first)) == [:link_matrix, :otherColumn]
    @test length(Dataset.buffer) == 1
    @test length(first.link_matrix) == 100
    @test length(first.otherColumn) == 1

    # Test getting a batch of data at once
    multiple=Dataset[[1, 45, 75, 142]]
    @test length(multiple) == 4
    @test length(multiple[1].link_matrix) == 100
    @test length(multiple[1].otherColumn) == 1
    @test length(Dataset.buffer) == 2
end

@testitem "DataLoader.getindex jld" tags=[:dataloader] setup=[MakeData] begin
    Dataset=QuantumGrav.DataLoader.Dataset(dir; cache_size = 2, mode = "jld2")

    # Test getting an index from the Dataset
    first=Dataset[1]

    @test collect(keys(first)) == [:link_matrix, :otherColumn]
    @test length(Dataset.buffer) == 1
    @test length(first.link_matrix) == 100
    @test length(first.otherColumn) == 1

    # Test getting a batch of data at once
    multiple=Dataset[[1, 45, 75, 142]]
    @test length(multiple) == 4
    @test length(multiple[1].link_matrix) == 100
    @test length(multiple[1].otherColumn) == 1
    @test length(Dataset.buffer) == 2
end
