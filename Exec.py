from sim_config import SimulationConfig, build_derived_quantities
from Bi_OU_core import run_simulation
from plot import plot_surface


cfg = SimulationConfig()
derived = build_derived_quantities(cfg)

V_x, V_y = run_simulation(cfg, derived)

plot_surface(V_x[0], title="Vx at t=0")
plot_surface(V_y[0], title="Vy at t=0")

