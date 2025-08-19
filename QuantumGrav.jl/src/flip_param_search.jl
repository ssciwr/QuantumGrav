include("csetmerging_MCMC.jl")

j = 1;

function minimize_flip_param(atom_count1::Int64, 
                                atom_count2::Int64, 
                                upper_right_connectivity_goal::Float64, 
                                runs_per_step::Int64,
                                initial_interval::Tuple{Float64,Float64}; 
                                initial_num_iterations::Int64 = 20,
                                rel_tol::Float64 = .01,
                                max_steps::Int64 = 20,
                                statistical_error::Float64 = 0.1)
    global j

    flip_param_down = initial_interval[1]
    flip_param_up = initial_interval[2]
    step = 1
    num_iterations = initial_num_iterations

    φ = (1 + sqrt(5)) / 2
    res_tab_1 = falses(runs_per_step)
    res_tab_2 = falses(runs_per_step)
  
    num_falses_1 = 0;
    num_falses_2 = 0;

    while step < max_steps
        flip_param_1 = flip_param_up - (flip_param_up - flip_param_down) / φ
        flip_param_2 = flip_param_down + (flip_param_up - flip_param_down) / φ

        sum_errors_1 = 0
        sum_errors_2 = 0

        for i in 1:runs_per_step
            cset1, _ = sample_bitarray_causet_by_connectivity(atom_count1, .5, 20, MersenneTwister(1234 + j); abs_tol = 0.01)
            j += 1
            cset2, _ = sample_bitarray_causet_by_connectivity(atom_count2, .3, 20, MersenneTwister(1234 + j); abs_tol = 0.01)
            j += 1

            res_1 = merge_csets_MCMC(flip_param_1, cset1, cset2, upper_right_connectivity_goal, num_iterations, MersenneTwister(1234 + j); abs_tol = .01)
            res_tab_1[i] = res_1[2]
            sum_errors_1 += res_1[3]
            j += 1

            res_2 = merge_csets_MCMC(flip_param_2, cset1, cset2, upper_right_connectivity_goal, num_iterations, MersenneTwister(1234 + j); abs_tol = .01)
            res_tab_2[i] = res_2[2]
            sum_errors_2 += res_2[3]
            j += 1
        end

        num_falses_1 = count(!, res_tab_1)
        num_falses_2 = count(!, res_tab_2)
        rel_diff = abs(num_falses_1 - num_falses_2) / max(num_falses_1, num_falses_2, 1)

        if rel_diff > statistical_error
            if num_falses_1 < num_falses_2
                flip_param_up = flip_param_2
            else
                flip_param_down = flip_param_1
            end
        else
            if sum_errors_1 < sum_errors_2
                flip_param_up = flip_param_2
            else
                flip_param_down = flip_param_1
            end
        end

        if all(res_tab_1) || all(res_tab_2)
            num_iterations = max(2, num_iterations - Int64(ceil(num_iterations / 4)))
        end
        
        if min(num_falses_1,num_falses_2) < 0.05 * num_iterations
            return flip_param_down, min(num_falses_1,num_falses_2)
        end

        #if abs(flip_param_up - flip_param_down) > rel_tol * (flip_param_up + flip_param_down) / 2
        #    return flip_param_down, min(num_falses_1,num_falses_2)
        #end

        step += 1
    end
    return flip_param_down, min(num_falses_1,num_falses_2)
end

test,_ = minimize_flip_param(130, 140, .3, 100, (0.0000001, 0.0005); statistical_error = .01, initial_num_iterations = 400)

0.002411194789657692

cset1, _ = sample_bitarray_causet_by_connectivity(200, .5, 20, MersenneTwister(1234 + j); abs_tol = 0.01)
j += 1
cset2, _ = sample_bitarray_causet_by_connectivity(210, .3, 20, MersenneTwister(1234 + j); abs_tol = 0.01)
j += 1
merge_csets_MCMC(0.0001, cset1, cset2, .3, 50, MersenneTwister(1234 + j); abs_tol = .01)
            res_tab_2[i] = res_2[2]

cset1.future_relations

cset_merged.future_relations

count_edges_upper_right_corner(cset, 2^6)


cset_merged.future_relations

count_edges_upper_right_corner(merged_cset,2)