import fast_tsp
import time
import os
import json
import random
import numpy as np

def tasks_to_dists(tasks):
    max_in_all_dist = 0
    dists_per_task = []
    setsize_per_task = []
    for task in tasks:
        filters = task["filters"]
        filters = [np.array(f) for f in filters]
        max_value = int(max([f.max() for f in filters])) + 1
        filters_bool = []
        for f in filters:
            f_bool = np.zeros(max_value, dtype=np.uint8)
            f_bool[f] = 1
            filters_bool.append(f_bool)

        setsize = 0
        for f in filters:
            setsize += f.sum()
        setsize_per_task.append(setsize)

        n = len(filters)
        dist = []
        for i in range(n):
            this_dists = []
            for j in range(n):
                # |filter_i xor filter_j|
                this_dists.append(int(np.sum(filters_bool[i] != filters_bool[j])))
                max_in_all_dist = max(max_in_all_dist, this_dists[-1])
            dist.append(this_dists)
        dists_per_task.append(dist)

    # import pdb; pdb.set_trace()
    return dists_per_task, setsize_per_task, max_in_all_dist

def load_all_tasks(file_path):
    lines = open(file_path, "r").readlines()
    tasks = []
    for line in lines:
        tasks.append(json.loads(line))
    return tasks_to_dists(tasks)

def create_larger_tasks(file_path, replicated_num, n_tasks):
    lines = open(file_path, "r").readlines()
    tasks = []
    i = 0
    for line in lines[:n_tasks]:
        print("load " + str(i))
        i += 1
        tasks.append(json.loads(line))
    lines = None

    # import pdb; pdb.set_trace()
    concated_tasks = []
    for i in range(0, len(tasks)):
        filters = []
        for j in range(replicated_num):
            filters.extend(tasks[(i+j) % len(tasks)]["filters"])
        concated_tasks.append({"filters": filters})

    return tasks_to_dists(concated_tasks), concated_tasks

def get_task_from_line(line):
    # import pdb; pdb.set_trace()
    task = json.loads(line)
    filters = task["filters"]
    # none if not provided
    camera_centers = task.get("camera_centers", None)
    key_dim = task.get("key_dim", None)

    dists_per_task, setsize_per_task, max_in_all_dist = tasks_to_dists([{"filters": filters}])

    dists = dists_per_task[0]
    setsize = setsize_per_task[0]
    return dists, setsize, max_in_all_dist, filters, camera_centers, key_dim

def compare_on_tsp_task(line, output_file, timeout_threshold, average_stats, all_evaluted_methods):

    dists, setsize, max_in_all_dist, filters, camera_centers, key_dim = get_task_from_line(line)
    downsample_ratio = max_in_all_dist // 30000 + 1

    downsampled_dist = []
    for i in range(len(dists)):
        downsampled_dist.append([])
        for j in range(len(dists[i])):
            downsampled_dist[-1].append(dists[i][j] // downsample_ratio)

    # nxn matrix
    print(f"Benchmarking on{len(dists)}x{len(dists)} matrix: {json.dumps(dists)}")
    output_file.write(f"Benchmarking on{len(dists)}x{len(dists)} matrix: {json.dumps(dists)}\n")

    for method in all_evaluted_methods:
        start_time = time.time()
        if method == "random":
            # a random permutation of n
            tour = random.sample(range(len(downsampled_dist)), len(downsampled_dist))
        elif method == "greedy_nearest_neighbor":
            tour = fast_tsp.greedy_nearest_neighbor(downsampled_dist)
        elif method == "greedy_sls":
            tour = fast_tsp.find_tour(downsampled_dist, timeout_threshold * 0.001)
        elif method == "optimal":
            if len(downsampled_dist) > 20:
                continue
            tour = fast_tsp.solve_tsp_exact(dists)
        elif method == "no_cache":
            cost = 0
            for i in range(len(downsampled_dist)):
                cost += len(filters[i])
        elif method == "gs_count_dec":
            tour = sorted(range(len(filters)), key=lambda x: len(filters[x]), reverse=True)
        elif method == "cam_position":
            if camera_centers is None or key_dim is None:
                raise ValueError("Camera centers or key dimension not provided for cam_position method.")
            tour = sorted(range(len(filters)), key=lambda x: camera_centers[x][key_dim])
        else:
            raise NotImplementedError

        # import pdb; pdb.set_trace()

        if method in ["random", "greedy_sls", "gs_count_dec", "cam_position"]:
            cost = len(filters[tour[0]])
            for i in range(1, len(tour)):
                # dists[tour[i-1]][tour[i]] = |A| + |B| - 2|A and B|
                # |B| - |A and B| = |B| - |A| + |A and B|
                cost += len(filters[tour[i]]) - (len(filters[tour[i-1]]) + len(filters[tour[i]]) - dists[tour[i-1]][tour[i]]) // 2

            # cost += len(filters[tour[-1]]) + len(filters[tour[0]])
            # cost = 2 * \sum |A| - 2 * \sum |A and B|

            # assert cost % 2 == 0, f"Cost is not even: {cost}"
            # cost //= 2

        elapsed_time = time.time() - start_time
        # cost = fast_tsp.compute_cost(tour, dists)
        print(f"Method: {method}, Cost: {cost}, Elapsed time: {elapsed_time}")

        output_file.write(f"Method: {method}, Cost: {cost}, Elapsed time: {elapsed_time}\n")
        average_stats[method].append(cost)


def update_average_stats(average_stats, n_gs):
    average_stats["naive_offload_volume"] = n_gs * 59 * 4 / (1024**3)
    average_stats["no_cache_volume"] = average_stats["no_cache"] * 48 * 4 / (1024**3)
    average_stats["random_volume"] = average_stats["random"] * 48 * 4 / (1024**3)
    average_stats["gs_count_dec_volume"] = average_stats["gs_count_dec"] * 48 * 4 / (1024**3)
    average_stats["cam_position_volume"] = average_stats["cam_position"] * 48 * 4 / (1024**3)
    average_stats["tsp_volume"] = average_stats["greedy_sls"] * 48 * 4 / (1024**3)

    average_stats["improve_over_naiveoffload"] = 1.0 - average_stats["tsp_volume"] / average_stats["naive_offload_volume"]
    average_stats["improve_over_random"] = 1.0 - average_stats["tsp_volume"] / average_stats["random_volume"]
    average_stats["improve_over_no_cache"] = 1.0 - average_stats["tsp_volume"] / average_stats["no_cache_volume"]
    average_stats["improve_over_gscountdec"] = 1.0 - average_stats["tsp_volume"] / average_stats["gs_count_dec_volume"]
    average_stats["improve_over_camposition"] = 1.0 - average_stats["tsp_volume"] / average_stats["cam_position_volume"]
    return average_stats
    

if __name__ == "__main__":

    file_path1 = "/home/hexu/Grendel-XS/output/_major-revision/bigcity/stat/20250718_030907_4090__4_bigcity_w=1_reorder_46mpcd_finalv7/sampled_filters.log"
    file_path2 = "/home/hexu/Grendel-XS/output/_major-revision/alameda/stat/20250718_030557_4090__4_alameda_reorder_28.6mpcd_finalv7/sampled_filters.log"
    file_path3 = "/home/hexu/Grendel-XS/output/_major-revision/ithaca/loc0/stat/20250718_031339_4090__4_ithacaloc0_reorder_40mpcd_finalv7/sampled_filters.log"
    file_path4 = "/home/hexu/Grendel-XS/output/_major-revision/rubble4k/stat/20250718_030137_4090__4_rubble4k_reorder_30mpcd_finalv7/sampled_filters.log"
    file_path5 = "/home/hexu/Grendel-XS/output/_major-revision/bicycle4k/stat/20250718_025551_4090__4_bicycle4k_reorder_9mpcd_finalv7/sampled_filters.log"
    num_gaussians_map = {
        file_path1: 46004112,
        file_path2: 28553559,
        file_path3: 40000016,
        file_path4: 30352516,
        file_path5: 9636072
    }
    # [102231360, 42830339, 76000030, 40397406, 9636072]

    all_stats = {}
    for file_path in [
        file_path5,
        file_path4,
        file_path2,
        file_path3,
        file_path1
    ]:
        scene_name = file_path.split("/")[6]
        print(f"Processing scene: {scene_name}")
        timeout_threshold = 1
        replicated = 1
        n_tasks = 30
        n_gs = num_gaussians_map[file_path]
        lines = open(file_path, "r").readlines()
        save_path = file_path.replace(".log", f"results_rep{replicated}_time{timeout_threshold}.log")

        if os.path.exists(save_path):
            # Average stats: {"random": 5242059.333333333, "greedy_sls": 5045556.766666667, "no_cache": 8177307.533333333, "gs_count_dec": 5074423.933333334}
            with open(save_path, "r") as f:
                lines = f.readlines()
                last_line = lines[-1]
                if "Average stats" in last_line:
                    average_stats = json.loads(last_line.split("Average stats: ")[-1])
                    average_stats = update_average_stats(average_stats, n_gs)
                    keys = list(average_stats.keys())
                    for key in keys:
                        average_stats[key] = round(average_stats[key], 2)
                    # print(f"Loaded average stats: {average_stats}")
                else:
                    print(f"File {save_path} does not contain average stats.")
            print(file_path)
            print(f"Average stats: {json.dumps(average_stats, indent=4)}")
        else:
            random.seed(42)
            np.random.seed(42)
            all_evaluated_methods = [
                "random",
                "greedy_sls",
                "no_cache",
                "gs_count_dec",
                "cam_position"
            ]
            new_stats = {method: [] for method in all_evaluated_methods}

            save_file = open(save_path, "w")
            for i in range(n_tasks):
                print(f"Processing Task {i}")
                compare_on_tsp_task(lines[i],
                    save_file,
                    timeout_threshold,
                    new_stats,
                    all_evaluted_methods=all_evaluated_methods
                )
            average_stats = {k: np.mean(v) for k, v in new_stats.items()}

            average_stats = update_average_stats(average_stats)
            save_file.write(f"Average stats: {json.dumps(average_stats)}")
            save_file.close()
        all_stats[scene_name] = average_stats
    print("All stats:")
    print(json.dumps(all_stats, indent=4))


















