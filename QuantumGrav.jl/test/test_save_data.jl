
@testitem "test_default_chunks" tags = [:save_data] begin
    import QuantumGrav
    import Zarr

    data = rand(Float32, 100, 200, 50)
    chunks = QuantumGrav.default_chunks(data)
    @test chunks == (16, 16, 16)

    data = rand(Float64, 10, 128)
    chunks = QuantumGrav.default_chunks(data)
    @test chunks == (10, 64)

    data = rand(Int64, 5000)
    chunks = QuantumGrav.default_chunks(data)
    @test chunks == (1024,)

    data = rand(Float32, 35, 3, 12, 5, 7)
    chunks = QuantumGrav.default_chunks(data)
    @test chunks == (8, 3, 8, 5, 7)
end



@testitem "test_write_arraylike" tags = [:save_data] begin
    import QuantumGrav
    import Zarr

    if isdir(joinpath(tempdir(), "test.zarr"))
        rm(joinpath(tempdir(), "test.zarr"), recursive=true)
    end

    file = Zarr.DirectoryStore(joinpath(tempdir(), "test.zarr"))
    group = Zarr.zgroup(file, "testgroup")

    dataarray = rand(Float64, 100, 101, 102)
    QuantumGrav.write_arraylike_to_zarr(group, "testdata", dataarray)

    @test isdir(joinpath(tempdir(), "test.zarr", "testgroup", "testdata"))
    arr = Zarr.zopen(file, "r"; path=joinpath("testgroup", "testdata"))

    print("metadata: ", arr.metadata.chunks)
    @test arr.metadata.chunks == (16, 16, 16)
    @test arr.metadata.shape[] == (100, 101, 102)
    @test arr[1:10, 1:3, 1:5] == dataarray[1:10, 1:3, 1:5]


    # custom chunking strat
    custom_chunking(x::AbstractArray) = (12, 15, 10)

    QuantumGrav.write_arraylike_to_zarr(
        group,
        "testdata_custom",
        dataarray;
        chunking_strategy=custom_chunking,
    )

    @test isdir(joinpath(tempdir(), "test.zarr", "testgroup", "testdata_custom"))

    arr = Zarr.zopen(file, "r"; path=joinpath("testgroup", "testdata_custom"))

    @test arr.metadata.chunks == (12, 15, 10)
    @test arr.metadata.shape[] == (100, 101, 102)
    @test arr[1:10, 1:3, 1:5] == dataarray[1:10, 1:3, 1:5]

    # no chunking
    QuantumGrav.write_arraylike_to_zarr(
        group,
        "testdata_custom_nochunk",
        dataarray;
        chunking_strategy = nothing,
    )

    @test isdir(joinpath(tempdir(), "test.zarr", "testgroup", "testdata_custom_nochunk"))

    arr = Zarr.zopen(file, "r"; path = joinpath("testgroup", "testdata_custom_nochunk"))

    @test arr.metadata.chunks == (12, 15, 10)
    @test arr.metadata.shape[] == (100, 101, 102)
    @test arr[1:10, 1:3, 1:5] == dataarray[1:10, 1:3, 1:5]
    rm(joinpath(tempdir(), "test.zarr"), recursive = true)
end


@testitem "dict_to_zarr" tags = [:save_data] begin
    import QuantumGrav
    import Zarr
    if isdir(joinpath(tempdir(), "dict_test.zarr"))
        rm(joinpath(tempdir(), "dict_test.zarr"), recursive=true)
    end

    if isdir(joinpath(tempdir(), "dict_test_customchunk.zarr"))
        rm(joinpath(tempdir(), "dict_test_customchunk.zarr"), recursive=true)
    end

    file = Zarr.DirectoryStore(joinpath(tempdir(), "dict_test.zarr"))
    root = Zarr.zgroup(file, "")
    data = Dict(
        "d1" => Dict("v1" => rand(Float32, 10, 12, 5), "i1" => rand(Int64, 20, 20)),
        "arr" => ["a", "bc", "c", "defeg"],
        "str" => "fj;aejfeiafhuaefhauefhafheausfasfeaf",
    )

    QuantumGrav.dict_to_zarr(root, data)

    @test isdir(joinpath(tempdir(), "dict_test.zarr", "d1", "v1"))
    @test isdir(joinpath(tempdir(), "dict_test.zarr", "d1", "i1"))
    @test isdir(joinpath(tempdir(), "dict_test.zarr", "arr"))
    @test isdir(joinpath(tempdir(), "dict_test.zarr", "str"))

    arr = Zarr.zopen(file, "r"; path=joinpath("d1", "v1"))
    @test arr.metadata.shape[] == (10, 12, 5)
    @test arr.metadata.chunks == (10, 12, 5)
    @test eltype(arr[:]) == Float32

    arr = Zarr.zopen(file, "r"; path=joinpath("d1", "i1"))
    @test arr.metadata.shape[] == (20, 20)
    @test arr.metadata.chunks == (20, 20)
    @test eltype(arr[:]) == Int64

    arr = Zarr.zopen(file, "r"; path="arr")
    @test arr.metadata.shape[] == (4,)
    @test arr.metadata.chunks == (4,)
    @test arr[:] == ["a", "bc", "c", "defeg"]

    arr = Zarr.zopen(file, "r"; path="str")
    @test arr.metadata.shape[] == (1,)
    @test arr.metadata.chunks == (1,)
    @test arr[:] == ["fj;aejfeiafhuaefhauefhafheausfasfeaf"]
    rm(joinpath(tempdir(), "dict_test.zarr"), recursive=true)

    chunking_strat = Dict(
        "v1" => x -> (5, 6, 5),
        "i1" => x -> (4, 4),
        "arr" => x -> (10,),
        "str" => x -> (7,),
    )

    file = Zarr.DirectoryStore(joinpath(tempdir(), "dict_test_customchunk.zarr"))
    root = Zarr.zgroup(file, "")
    QuantumGrav.dict_to_zarr(root, data; chunking_strategy=chunking_strat)

    @test isdir(joinpath(tempdir(), "dict_test_customchunk.zarr", "d1", "v1"))
    @test isdir(joinpath(tempdir(), "dict_test_customchunk.zarr", "d1", "i1"))
    @test isdir(joinpath(tempdir(), "dict_test_customchunk.zarr", "arr"))
    @test isdir(joinpath(tempdir(), "dict_test_customchunk.zarr", "str"))


    arr = Zarr.zopen(file, "r"; path=joinpath("d1", "v1"))
    @test arr.metadata.shape[] == (10, 12, 5)
    @test arr.metadata.chunks == (5, 6, 5)
    @test eltype(arr[:]) == Float32

    arr = Zarr.zopen(file, "r"; path=joinpath("d1", "i1"))
    @test arr.metadata.shape[] == (20, 20)
    @test arr.metadata.chunks == (4, 4)
    @test eltype(arr[:]) == Int64

    arr = Zarr.zopen(file, "r"; path="arr")
    @test arr.metadata.shape[] == (4,)
    @test arr.metadata.chunks == (10,)
    @test arr[:] == ["a", "bc", "c", "defeg"]

    arr = Zarr.zopen(file, "r"; path="str")
    @test arr.metadata.shape[] == (1,)
    @test arr.metadata.chunks == (7,)
    @test arr[:] == ["fj;aejfeiafhuaefhauefhafheausfasfeaf"]
    rm(joinpath(tempdir(), "dict_test_customchunk.zarr"), recursive=true)

end
