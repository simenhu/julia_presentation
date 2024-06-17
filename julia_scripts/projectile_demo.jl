using Plots
using Distributed
using Zygote
using Measurements

# Physical constants
const g = 9.81  # Gravity (m/s^2)
const c = 0.1   # Air resistance coefficient
const dt = 0.01 # Time step

# Step 1: Single-threaded Calculation
function euler_step(u, du, dt)
    return u .+ dt .* du
end

function projectile_dynamics(u)
    x, y, vx, vy = u
    du = [vx, vy, -c * vx, -g - c * vy]
    return du
end

function simulate_projectile(angle, velocity, tspan=(0.0, 10.0))
    u = [0.0, 0.0, velocity * cos(angle), velocity * sin(angle)]
    trajectory = [copy(u)]
    t = tspan[1]

    while t < tspan[2] && u[2] >= 0  # Stop if the projectile hits the ground
        du = projectile_dynamics(u)
        u = euler_step(u, du, dt)
        trajectory = vcat(trajectory, [copy(u)])  # Concatenate arrays instead of using push!
        t += dt
    end
    
    return trajectory
end

function extract_range(trajectory)
    return maximum(map(x -> x[1], trajectory))
end

angle = π/4
velocity = 20.0
trajectory = simulate_projectile(angle, velocity)
plot(map(x -> x[1], trajectory), map(x -> x[2], trajectory), xlabel="x", ylabel="y", label="Trajectory")
println("Range: ", extract_range(trajectory))

# Step 2: Multiprocessing
addprocs(4)

@everywhere begin
    using Plots
    
    const g = 9.81
    const c = 0.1
    const dt = 0.01
    
    function euler_step(u, du, dt)
        return u .+ dt .* du
    end

    function projectile_dynamics(u)
        x, y, vx, vy = u
        du = [vx, vy, -c * vx, -g - c * vy]
        return du
    end

    function simulate_projectile(angle, velocity, tspan=(0.0, 10.0))
        u = [0.0, 0.0, velocity * cos(angle), velocity * sin(angle)]
        trajectory = [copy(u)]
        t = tspan[1]

        while t < tspan[2] && u[2] >= 0  # Stop if the projectile hits the ground
            du = projectile_dynamics(u)
            u = euler_step(u, du, dt)
            trajectory = vcat(trajectory, [copy(u)])  # Concatenate arrays instead of using push!
            t += dt
        end

        return trajectory
    end

    function extract_range(trajectory)
        return maximum(map(x -> x[1], trajectory))
    end
end

angles = range(0, stop=π/2, length=10)
@distributed for i in 1:length(angles)
    trajectory = simulate_projectile(angles[i], velocity)
    println("Range at angle $(angles[i]): ", extract_range(trajectory))
end

# Step 3: Automatic Differentiation
range_func(angle, velocity) = extract_range(simulate_projectile(angle, velocity))
gradient = Zygote.gradient(range_func, angle, velocity)
println("Gradient: ", gradient)

# Step 4: Uncertainty Quantification
angle_uncertain = Measurement(π/4, 0.1)
velocity_uncertain = Measurement(20.0, 1.0)

trajectory_uncertain = simulate_projectile(angle_uncertain, velocity_uncertain)
range_uncertain = extract_range(trajectory_uncertain)
println("Range with uncertainty: ", range_uncertain)
println("Uncertainty in the range: ", uncertainty(range_uncertain))
