import logging
import sys
import gamspy as gp
import os
import time

logging.disable(logging.WARNING)


def solve_lqcp(solver, N):
    container = gp.Container()
    with container:
        n = N
        m = N
        dx = 1.0 / n
        T = 1.58
        dt = T / n
        h2 = dx**2
        a = 0.001
        ns = gp.Set(records=range(n + 1))
        ms = gp.Alias(alias_with=ns)
        yt = gp.Parameter(domain=ns)

        y = gp.Variable(domain=[ms, ns])
        y.lo = 0
        y.up = 1
        u = gp.Variable(domain=ms)
        u.lo = -1.0
        u.up = 1.0
        yt[ns] = 0.5 * (1 - (ns.val * dx) ** 2)

        pde = gp.Equation(domain=[ns, ms])
        ic = gp.Equation(domain=ns)
        bc1 = gp.Equation(domain=ns)
        bc2 = gp.Equation(domain=ns)

        obj = (
            1 / 4 * dx * (y[str(m), "0"] - yt["0"]) ** 2
            + 2 * gp.Sum(ns.where[gp.Ord(ns) > 1], (y[str(m), ns] - yt[ns]) ** 2)
            + (y[str(m), str(n)] - yt[str(n)]) ** 2
            + 1
            / 4
            * a
            * dt
            * (2 * gp.Sum(ms.where[gp.Ord(ms) > 1], u[ms] ** 2) + u[str(m)] ** 2)
        )

        pde[ns, ms].where[(gp.Ord(ns) < gp.Card(ns)) & (gp.Ord(ms) > 1) & (gp.Ord(ms) < gp.Card(ms))] = (
            y[ns + 1, ms] - y[ns, ms]
        ) / dt == 0.5 * (
            y[ns, ms - 1]
            - 2 * y[ns, ms]
            + y[ns, ms + 1]
            + y[ns + 1, ms - 1]
            - 2 * y[ns + 1, ms]
            + y[ns + 1, ms + 1]
        ) / h2
        ic[ns] = y["0", ns] == 0
        bc1[ns].where[(gp.Ord(ns) > 1) & (gp.Ord(ns) < gp.Card(ns))] = (
            y[ns, "2"] - 4 * y[ns, "1"] + 3 * y[ns, "0"] == 0
        )
        bc2[ns].where[(gp.Ord(ns) > 1) & (gp.Ord(ns) < gp.Card(ns))] = y[ns, str(n - 2)] - 4 * y[
            ns, str(n - 1)
        ] + 3 * y[ns, str(n)] == (2 * dx) * (u[ns] - y[ns, str(n)])

        model = gp.Model(
            name="lqcp",
            equations=container.getEquations(),
            problem=gp.Problem.MIQCP,
            sense=gp.Sense.MIN,
            objective=obj,
        )
        model.solve(solver=solver, options=gp.Options(time_limit=0))


def main(Ns=[500, 1000, 1500, 2000]):
    solver = sys.argv[1]
    if solver not in ("gurobi", "copt"):
        raise ValueError(f"Unknown solver {solver}.")

    dir = os.path.realpath(os.path.dirname(__file__))
    for n in Ns:
        start = time.time()
        _ = solve_lqcp(solver, n)
        run_time = round(time.time() - start, 1)
        content = f"gamspy lqcp-{n} -1 {run_time}"
        print(content)
        with open(dir + "/benchmarks.csv", "a") as io:
            io.write(f"{content}\n")
    return


main()
