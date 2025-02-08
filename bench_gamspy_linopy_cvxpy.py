import sys
import time
import numpy as np
import pandas as pd
import gamspy as gp


def solve_gamspy_model(N: int, solver: str) -> None:
    m = gp.Container()
    with m:
        i = gp.Set(records=range(N))
        j = gp.Alias(alias_with=i)
        a = gp.Parameter(domain=i, records=np.arange(N))
        x = gp.Variable(domain=[i, j])
        y = gp.Variable(domain=[i, j])
        eq1 = gp.Equation(domain=[i, j])
        eq1[i, j] = x[i, j] - y[i, j] >= a[i]
        eq2 = gp.Equation(domain=[i, j])
        eq2[i, j] = x[i, j] + y[i, j] >= 0
        obj = 2 * gp.Sum((i, j), x[i, j]) + gp.Sum((i, j), y[i, j])
        model = gp.Model(
            name="bench",
            equations=m.getEquations(),
            sense=gp.Sense.MIN,
            problem=gp.Problem.LP,
            objective=obj,
        )
        m.addGamsCode("bench.justscrdir = 0")
        model.solve(solver=solver, options=gp.Options(time_limit=0))


def bench(N, solver_name):
    results = {}

    t0 = time.time()
    solve_gamspy_model(N, solver_name)
    t1 = time.time()
    results["gamspy"] = t1 - t0

    return results


def main(solver_name="gurobi"):
    Ns = range(100, 501, 100)
    results = []
    for N in Ns:
        results.append(bench(N, solver_name))
    # create a DataFrame
    df = pd.DataFrame(results, index=Ns)

    # show result
    print(df)


if __name__ == "__main__":
    # solver_name can be "copt", "gurobi", "highs"
    solver = sys.argv[1]
    if solver not in ("copt", "gurobi", "highs"):
        raise ValueError(f"Solver must be copt, gurobi or highs but given {solver}")
    main(solver)
