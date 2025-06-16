from dataclasses import dataclass
from typing import Callable, Optional
import pyswarms as ps
import numpy as np
from cluster_selection import ClusteringModule


def rosenbrock(x):
    """Calculates the Rosenbrock function value."""
    cost = sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
               for i in range(len(x)-1))
    return cost


@dataclass
class PsoOptimizator:
    n_iters: int
    n_particles: int
    dimensions: int
    fitness_func: Callable 
    min_bounds: list
    max_bounds: list
    options: tuple = (0.729, 1.49445, 1.49445) # w, c1 and c2
    cluster_module: Optional[ClusteringModule] = None

    def calc_fitness_solutions(self, solutions):
        """
        Calculate the fitness value for all
        candidate solutions. """
        costs = []

        for solution in solutions:
            # Prepare candidate solution for fitness function
            if self.cluster_module is not None:
                clusters, n_clusters = self.convert_solution_to_clusters( solution )

                cost = self.fitness_func(clusters, n_clusters)
            else:
                cost = self.fitness_func(solution)

            costs.append(cost)

        # get_DBC_distances([0, 1, 2, 1, 0], n_clusters)

        return costs

    def convert_solution_to_clusters(self, solution):
        dims_dataset = self.cluster_module.X.shape[1]
        n_clusters = len(solution) // dims_dataset

        centroids = solution.reshape((n_clusters, dims_dataset))

        clusters = self.cluster_module.get_clusters_by_centroids(centroids)
        return clusters, n_clusters
        

    def optimize(self):
        # pygad instance
        options = {'w': self.options[0], 'c1': self.options[1],
                   'c2': self.options[2]}

        self.bounds = (self.min_bounds, self.max_bounds)
        pso = ps.single.GlobalBestPSO(
            n_particles=self.n_particles, dimensions=self.dimensions,
            options=options, bounds=self.bounds
        )
        best_cost, best_solution = pso.optimize(
            self.calc_fitness_solutions, iters=self.n_iters)


        if self.cluster_module is not None:
            clusters, _ = self.convert_solution_to_clusters(best_solution)
            return clusters, best_cost

        return best_solution, best_cost

if __name__ == "__main__":
    n_iters = 30
    n_particles = 100
    dimensions = 20
    min_bounds = [-1] * dimensions
    max_bounds = [ 1] * dimensions

    ev_optim = PsoOptimizator(
        n_iters, n_particles, dimensions, rosenbrock, min_bounds, max_bounds
    )
    ev_optim.optimize()
