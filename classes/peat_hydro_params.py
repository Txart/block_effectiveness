class PeatlandHydroParameters:
    def __init__(self, dt, dx, nx, ny,
                 max_sweeps, fipy_desired_residual,
                 s1, s2, t1, t2,
                 use_several_weather_stations) -> None:
        self.dt = dt
        self.dx = dx
        self.nx = nx
        self.ny = ny
        self.max_sweeps = max_sweeps
        self.fipy_desired_residual = fipy_desired_residual

        # parameters for S and T functions. Also, theta to zeta
        self.s1 = s1
        self.s2 = s2
        self.t1 = t1
        self.t2 = t2

        # Compute P-ET based on weighted average distance of several weather stations
        self.use_several_weather_stations = use_several_weather_stations

        pass