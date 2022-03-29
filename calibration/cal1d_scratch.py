# %% Code to check calibration_1d.py quantities
name = 'RA'
NDAYS = 10
s2, t1, t2 =[0.1,  1, 20]

heights = np.array(dipwell_heights[name]) + EXTRA_PEAT_DEPTH

dem = transect_dem[name]
df = data_dict[name]
        
ZETA_LEFT_DIRIS, zeta_middle_measurements, ZETA_RIGHT_DIRIS = get_sensor_measurements(df=df, t_name=name)
source = df['source'].to_numpy()

NDAYS = len(ZETA_LEFT_DIRIS) - 1 # First datapoint is for initial condition

zeta_ini = initial_zetas[name]

h_ini = h_from_zeta(zeta_ini, dem)
theta = theta_from_h(h_ini, s2)

SOURCE = source[:-1]
ZETA_LEFT_DIRIS = ZETA_LEFT_DIRIS[:-1]
ZETA_RIGHT_DIRIS = ZETA_RIGHT_DIRIS[:-1]

solution_zeta = solve_transient_both_diri(theta=theta,
                                    dx=dx, dt=dt, NDAYS=NDAYS, N=NX,
                                    dem=dem, s2=s2, t1=t1, t2=t2, SOURCE=SOURCE,
                                    ZETA_LEFT_DIRIS=ZETA_LEFT_DIRIS, ZETA_RIGHT_DIRIS=ZETA_RIGHT_DIRIS,
                                    MAX_INTERNAL_NITER=MAX_INTERNAL_NITER, WEIGHT=WEIGHT,
                                    REL_TOL=REL_TOL, ABS_TOL=ABS_TOL, verbose=False)

plt.figure()
plt.title('SOURCE')
plt.plot(source)

plt.figure()
plt.title('INI COND')
plt.plot(MESH, h_ini, color='green', label='measured')


zeta_sol_middle_sensors_all_days = np.zeros(shape=(NDAYS, zeta_middle_measurements.shape[1]))
all_sensor_measurements = df[[name + '.0', name + '.1', name + '.2', name + '.3', name + '.4', name + '.5']].to_numpy()
for day in range(NDAYS):
    plt.figure()
    plt.title(day)
    zeta_sol_interp_func = interp1d(x=MESH, y=solution_zeta[day])
    zeta_sol_middle_sensors = zeta_sol_interp_func(dipwell_separation[1:-1]) # first and last sensors are BC
    zeta_sol_middle_sensors_all_days[day] = zeta_sol_middle_sensors

    plt.plot(MESH, h_from_zeta(solution_zeta[day], dem), 'o', label='solution', color='blue')
    plt.plot(dipwell_separation[1:-1], h_from_zeta(zeta_sol_middle_sensors, heights[1:-1]), 'o', color='red', label='solution interpolated')
    plt.plot(dipwell_separation, h_from_zeta(all_sensor_measurements[day], heights), color='green', label='measured')
    plt.legend()




#%% plot all initial zetas

for name, zeta_ini in initial_zetas.items():
    plt.figure()
    plt.title(name)
    plt.plot(MESH, h_from_zeta(zeta_ini, transect_dem[name]), label='initial h')
    plt.plot(MESH, transect_dem[name], label='dem')
    plt.legend()
# %%
