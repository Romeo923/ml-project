from HopfieldNetwork import HopfieldNetwork, sigmoid

rows = 3
cols = 3
prob = 0.5
num_patterns = 3
num_cells = rows * cols

hop_res = (
    HopfieldNetwork(num_cells)
    .train_random_patterns(rows, cols, num_patterns, prob)
    .run_simulation(
        instances=3, steps=5, rows=rows, cols=cols, activ_func=sigmoid(beta=10)
    )
    .sim_result()
)

sim_ops = hop_res["ordered_params"]

print(sim_ops)
print()
# print(sim_ops[0])
# print()
# print(sim_ops[0][0])
# print()
first_ops = [sim_ops[sim][0] for sim in range(3)]
print(first_ops)
