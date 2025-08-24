module Parameters

export LBMParams, read_params_input




struct LBMParams
    # geometry / grid (physical lengths in same units as h)
    Nx::Int
    Ny::Int
    Nz::Int
    h::Float32

    # geometry references (physical)
    R::Float32
    x_c::Float32
    y_c::Float32
    theta0::Float32               # radians

    # lattice / flow
    q::Int
    Re::Float32
    U_inf::Float32                # lattice speed (cells/step)
    V_inf::Float32                # lattice speed (cells/step)
    W_inf::Float32                # lattice speed (cells/step)
    rho_inf::Float32
    dt::Float32                   # physical time per step

    # iteration control
    n_max::Int
    n_save::Int

    # derived transport / TRT (ALL lattice-units)
    nu::Float32                   # ν_lat
    omega_p::Float32
    omega_m::Float32
    Lambda::Float32

    # BC / collision labels
    bc::String
    coll::String

    # shape handling
    shape::Symbol                     # :CYL or :AIR
    naca_code::Union{Nothing,String}  # e.g. "0012"
    chord::Float32                    # physical length

    function LBMParams(Nx::Int, Ny::Int, Nz::Int,
                       h::Float32, R::Float32, x_c::Float32, y_c::Float32, theta0::Float32,
                       q::Int, Re::Float32,
                       U_inf::Float32, V_inf::Float32, W_inf::Float32,
                       rho_inf::Float32, dt::Float32,
                       Lambda::Float32,
                       n_max::Int, n_save::Int,
                       bc::String, coll::String,
                       shape::Symbol, naca_code::Union{Nothing,String},
                       chord::Float32)

        # Characteristic length in LATTICE CELLS (Option A)
        # (R, chord are physical lengths in same units as h)
        L_char_lat = ((shape === :CYL) ? (2f0 * R) : chord) / h

        # Lattice viscosity from Re: Re = U_lat * L_char_lat / ν_lat
        # IMPORTANT: U_inf is lattice speed (cells/step)
        nu_lat = (U_inf * L_char_lat) / Re

        # TRT in lattice units (cs2 = 1/3)
        tau_p   = 0.5f0 + 3f0 * nu_lat
        tau_m   = 0.5f0 + Lambda / (tau_p - 0.5f0)
        omega_p = 1f0 / tau_p
        omega_m = 1f0 / tau_m

        return new(Nx, Ny, Nz, h,
                   R, x_c, y_c, theta0,
                   q, Re, U_inf, V_inf, W_inf, rho_inf, dt,
                   n_max, n_save,
                   nu_lat, omega_p, omega_m, Lambda,
                   bc, coll,
                   shape, naca_code, chord)
    end
end




function read_params_input(path::String)::Parameters.LBMParams
    lines = readlines(path)

    values = Float32[]
    bc::Union{Nothing,String}    = nothing
    coll::Union{Nothing,String}  = nothing
    shape_sym::Symbol            = :CYL
    naca_code::Union{Nothing,String} = nothing

    for raw in lines
        line = strip(raw)
        isempty(line)        && continue
        startswith(line, "#") && continue

        # String keys (order-independent)
        if startswith(line, "BC:")
            m = match(r"^BC:\s*\|?\s*([A-Za-z0-9_+\-]+)", line)
            @assert m !== nothing "Could not parse BC from: $raw"
            bc = String(m.captures[1]); continue
        elseif startswith(line, "Coll:")
            m = match(r"^Coll:\s*\|?\s*([A-Za-z0-9_+\-]+)", line)
            @assert m !== nothing "Could not parse Coll from: $raw"
            coll = String(m.captures[1]); continue
        elseif startswith(line, "Shape:")
            m = match(r"^Shape:\s*\|?\s*([A-Za-z]+)", line)
            @assert m !== nothing "Could not parse Shape from: $raw"
            sh = uppercase(String(m.captures[1]))
            @assert sh == "CYL" || sh == "AIR" "Shape must be CYL or AIR, got $sh"
            shape_sym = (sh == "CYL") ? :CYL : :AIR
            continue
        elseif startswith(line, "NACA:")
            m = match(r"^NACA:\s*\|?\s*([0-9]{4})", line)
            @assert m !== nothing "Could not parse NACA code (expect 4 digits) from: $raw"
            naca_code = String(m.captures[1]); continue
        end

        # Numeric values (first number on the line)
        m = match(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?", line)
        @assert m !== nothing "Could not find a number in: $raw"
        push!(values, parse(Float32, m.match))
    end

    # Numeric order expected (19 values):
    # Nx Ny Nz h R x_c y_c theta0 Chord q Re U_inf V_inf W_inf rho_inf dt Lambda n_max n_save
    @assert length(values) == 19 "Expected 19 numeric parameters, got $(length(values))."
    @assert bc !== nothing "Missing BC line in input file!"
    coll === nothing && (coll = "BGK")
    if shape_sym === :AIR && naca_code === nothing
        println("\nNo NACA input, falling back to default: NACA0012.\n")
        naca_code = "0012"
    end

    return Parameters.LBMParams(
        Int(values[1]),    # Nx
        Int(values[2]),    # Ny
        Int(values[3]),    # Nz
        values[4],         # h (physical length per cell)
        values[5],         # R (physical)
        values[6],         # x_c (physical)
        values[7],         # y_c (physical)
        deg2rad(values[8]),# # theta0 (radians)
        Int(values[10]),   # q
        values[11],        # Re
        values[12],        # U_inf (LATTICE, cells/step)
        values[13],        # V_inf (LATTICE)
        values[14],        # W_inf (LATTICE)
        values[15],        # rho_inf (lattice)
        values[16],        # dt (physical time per step)
        values[17],        # Lambda
        Int(values[18]),   # n_max
        Int(values[19]),   # n_save
        bc,
        coll,
        shape_sym,
        naca_code,
        values[9],         # chord (physical)
    )
end




end # module
