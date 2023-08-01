import run_experiment as bodi
import sys
import Config as cf
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1].split('=')[-1] != 'aruba':
        print('Please provide:',
              '\n \t testbed name: testbed= (Testbed1, Testbed2, aruba (this is a realworld dataset))',
              '\n \t (only for Testbed1 and Testbed2) ---> epsilon value: epsilon= (0.25, 0.5, 1)')
        sys.exit(1)

    if len(sys.argv) == 2 and sys.argv[1].split('=')[-1] != 'aruba':
        print('Please provide:',
              '\n \t testbed name: testbed= (Testbed1, Testbed2, aruba (this is a realworld dataset))',
              '\n \t (only for Testbed1 and Testbed2) ---> epsilon value: epsilon= (0.25, 0.5, 1)')
        sys.exit(1)


    evalfn = sys.argv[1].split('=')[-1]

    def is_valid(sensor_placeholder):
        # This is for checking locations where placing sensors are not allowed. 
        if cf.testbed == 'Testbed2/' and sensor_placeholder[0] <= 2 and sensor_placeholder[1] <= 2:
            return False
        else:
            return True
    def frange(X, Y):
        x = X
        y = Y

        W = []
        start = cf.epsilon

        while start < x:
            W.append(start)
            start += cf.epsilon

        H = []
        start = cf.epsilon

        while start < y:
            H.append(start)
            start += cf.epsilon

        grid = []

        for w in W:
            for h in H:
                if is_valid([w, h]):
                    grid.append([w, h])

        return grid

    cf.testbed = evalfn + '/'

    if evalfn == 'aruba':
        sensors = ['M00' + str(i) if i < 10 else 'M0' + str(i) for i in range(1,32)]
        num_items = len(sensors)
    else:
        if evalfn == 'Testbed1':
            epsilon = float(sys.argv[2].split('=')[-1])
            cf.epsilon = epsilon

            sensors = frange(8, 8)
            num_items = len(sensors)

        elif evalfn == 'Testbed2':
            epsilon = float(sys.argv[2].split('=')[-1])
            cf.epsilon = epsilon

            sensors = frange(8, 5.3)
            num_items = len(sensors)    

    Xs, Ys, meta_data = bodi.run_experiment(n_replications = 5,
                        save_to_pickle = True,
                        evalfn = evalfn,
                        n_initial_points = 3,
                        max_evals = 1000,
                        batch_size = 10,
                        n_binary = num_items)

    print('--------------')
    print(Xs)
    print('--------------')
    print(Ys)
    print('--------------')
    print(meta_data)

