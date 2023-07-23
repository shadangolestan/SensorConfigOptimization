import run_experiment as bodi

motion_places = ['M00' + str(i) if i < 10 else 'M0' + str(i) for i in range(1,32)]

Xs, Ys, meta_data = bodi.run_experiment(n_replications = 5,
                    save_to_pickle = True,
                    evalfn = 'aruba',
                    n_initial_points = 3,
                    max_evals = 1000,
                    batch_size = 10,
                    n_binary = len(motion_places))

print('--------------')
print(Xs)
print('--------------')
print(Ys)
print('--------------')
print(meta_data)

