module Motion

using ..Parameters
using ..Geometry
using ..SolidBC
using ..Initialization
using CUDA

export motion!, MotionLaw, motion_at




struct MotionLaw{Fx,Fxd,Fy,Fyd,Fth,Fthd}
    x::Fx
    xdot::Fxd
    y::Fy
    ydot::Fyd
    theta::Fth
    thetadot::Fthd
end




@inline function motion_at(n::Int, dt::Float32, m::MotionLaw; offset::Float32=0f0)
    t = (Float32(n - 1) + offset) * dt
    return (
        x_c   = Float32(m.x(t)),
        y_c   = Float32(m.y(t)),
        theta = Float32(m.theta(t)),
        U_c   = Float32(m.xdot(t)),
        V_c   = Float32(m.ydot(t)),
        omega = Float32(m.thetadot(t)),
    )
end




function motion!(
    x_lli::CuArray{Float32},
    y_lli::CuArray{Float32},
    nbr_lli::CuArray{Int32,2},       # (q, N_owned)

    state::SolidMaskState,
    mask_s_local::CuArray{Bool},     # length N_local
    mask_bb_local::CuArray{Bool},    # length N_owned
    cnt::CuArray{Int32},             # length N_owned (scratch)

    R::Float32, x_c::Float32, y_c::Float32,
    dt::Float32, h::Float32,
    bcstruct::SolidBCStruct,
    law::MotionLaw,
    n::Int, rank::Int
)
    if rank == 0
        return
    end

    # 0) current rigid-body pose/vel
    pose_n    = motion_at(n, dt, law; offset=0f0)       # for masks
    pose_half = motion_at(n, dt, law; offset=0.5f0)     # for moving-wall BC

    x_c  = pose_n.x_c;   y_c  = pose_n.y_c;   theta = pose_n.theta
    U_cH = pose_half.U_c; V_cH = pose_half.V_c; ωH = pose_half.omega
    x_cH = pose_half.x_c; y_cH = pose_half.y_c


    # 1) Generate local subdomain solid mask
    cθ = cos(theta); sθ = sin(theta)
    generate_mask_s_local!(
        mask_s_local,
        x_lli, y_lli,
        x_c, y_c,
        cθ, sθ,
        state
    )

    # 2) bb-node flags (owned)
    generate_mask_bb_local!(mask_bb_local, mask_s_local, nbr_lli)

    # 3) ensure capacity (upper bound nnz <= q * N_bb)
    q    = Int(size(nbr_lli, 1))
    N_bb = Int(CUDA.sum(mask_bb_local))          # CUDA supports sum(::CuArray)
    needed = q * N_bb
    ensure_bb_capacity!(bcstruct, rank, needed; slack=1.2, growth=2)

    # 4) build CSR on GPU using bcstruct fields
    nnz = compile_bb_data!(
        bcstruct.bb_dirs, bcstruct.bb_lidxs, bcstruct.bb_ptr, cnt,
        mask_s_local, mask_bb_local, nbr_lli, bcstruct
    )

    # 5) Update half-time kinematics (lattice units)
    bcstruct.U_c_lat = U_cH * dt / h
    bcstruct.V_c_lat = V_cH * dt / h
    bcstruct.ω_lat   = ωH   * dt
    bcstruct.x_c_lat = x_cH / h
    bcstruct.y_c_lat = y_cH / h

    # Neighbor map used by IBB (keep a handle in the struct)
    bcstruct.nbr_lli = nbr_lli

    # Geometry & orientation needed by IBB
    # (state.* here are the same values you already pass to generate_mask_s_local!)
    bcstruct.shape_flag       = state.shape_flag
    bcstruct.R_lat            = R / h
    bcstruct.chord_lat        = state.chord / h
    bcstruct.x_le_offset_lat  = state.x_le_offset / h
    bcstruct.m                = state.m
    bcstruct.p                = state.p
    bcstruct.t                = state.t
    bcstruct.x_int            = state.x_int

    # Half-time orientation for φ (IBB uses mid-time)
    θH = pose_half.theta
    bcstruct.cθ_half = cos(θH)
    bcstruct.sθ_half = sin(θH)

    return
end




end # module