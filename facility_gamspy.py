import sys
import gamspy as gp
import os
import time
import logging

logging.disable(logging.WARNING)


def solve_facility(solver, G, F):
    M = 2 * 1.414
    m = gp.Container()
    with m:
        grid = gp.Set(name="grid", records=range(G + 1))
        grid2 = gp.Alias(name="grid2", alias_with=grid)
        facs = gp.Set(name="facs", records=range(1, F + 1))
        dims = gp.Set(name="dims", records=range(1, 3))
        y = gp.Variable(name="y", domain=[facs, dims])
        y.lo = 0
        y.up = 1
        s = gp.Variable(name="s", domain=[grid, grid2, facs])
        s.lo = 0
        z = gp.Variable(
            name="z", domain=[grid, grid2, facs], type=gp.VariableType.BINARY
        )
        r = gp.Variable(name="r", domain=[grid, grid2, facs, dims])
        d = gp.Variable(name="d")

        assmt = gp.Equation(domain=[grid, grid2])
        assmt[...] = gp.Sum(facs, z[grid, grid2, facs]) == 1

        quadrhs = gp.Equation(domain=[grid, grid2, facs])
        quadrhs[...] = s[grid, grid2, facs] == d + M * (1 - z[grid, grid2, facs])

        quaddistk1 = gp.Equation(name="quaddistk1", domain=[grid, grid2, facs])
        quaddistk1[...] = r[grid, grid2, facs, "1"] == (1 * grid.val) / G - y[facs, "1"]

        quaddistk2 = gp.Equation(domain=[grid, grid2, facs])
        quaddistk2[...] = (
            r[grid, grid2, facs, "2"] == (1 * grid2.val) / G - y[facs, "2"]
        )

        quaddist = gp.Equation(domain=[grid, grid2, facs])
        quaddist[...] = (
            r[grid, grid2, facs, "1"] ** 2 + r[grid, grid2, facs, "2"]
            <= s[grid, grid2, facs] ** 2
        )

        model = gp.Model(
            name="facility",
            equations=m.getEquations(),
            problem=gp.Problem.MIQCP,
            sense=gp.Sense.MIN,
            objective=d,
        )
        m.addGamsCode("facility.justscrdir = 0;")
        model.solve(solver=solver, options=gp.Options(time_limit=0))


def main(Ns=[25, 50, 75, 100]):
    solver = sys.argv[1]
    if solver not in ("gurobi", "copt"):
        raise ValueError(f"Unknown solver {solver}.")

    dir = os.path.realpath(os.path.dirname(__file__))
    for n in Ns:
        start = time.time()
        _ = solve_facility(solver, n, n)
        run_time = round(time.time() - start, 1)
        content = f"gamspy fac-{n} -1 {run_time}"
        print(content)
        with open(dir + "/benchmarks.csv", "a") as io:
            io.write(f"{content}\n")
    return


main()
