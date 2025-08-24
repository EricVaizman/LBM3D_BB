module SolidBC

using CUDA
using ..Parameters
using ..Geometry
using ..Collision: INV_CS2

export solid_BC!, init_SolidBCStruct, SolidBCStruct




Base.@kwdef mutable struct SolidBCStruct
    # --- SBB / link list ---
    bb_lidxs::Union{Nothing,CuArray{Int32,1}} = nothing
    bb_dirs ::Union{Nothing,CuArray{Int8,1}}  = nothing
    bb_ptr  ::Union{Nothing,CuArray{Int32,1}} = nothing
    bbOpp   ::Union{Nothing,CuArray{Int8,1}}  = nothing
    nnz     ::Int32                            = 0

    # --- half-time kinematics (planar) in lattice units ---
    U_c_lat::Float32 = 0f0
    V_c_lat::Float32 = 0f0
    ω_lat  ::Float32 = 0f0
    x_c_lat::Float32 = 0f0
    y_c_lat::Float32 = 0f0

    # --- IBB extras ---
    nbr_lli::Union{Nothing,CuArray{Int32,2}} = nothing   # (Q, N_owned)
    shape_flag::Int8      = Int8(1)    # 1=cylinder, else airfoil
    R_lat::Float32        = 0f0        # cylinder radius (lat)
    chord_lat::Float32    = 0f0        # airfoil chord (lat)
    m::Float32            = 0f0
    p::Float32            = 0f0
    t::Float32            = 0f0
    x_int::Float32        = 1f0
    x_le_offset_lat::Float32 = 0f0     # LE offset (lat)
    cθ_half::Float32      = 1f0        # cos(theta_{n+1/2})
    sθ_half::Float32      = 0f0        # sin(theta_{n+1/2})

    # optional (unused if q is on-the-fly)
    q_arr::Union{Nothing,CuArray{Float32,1}} = nothing
end




function init_SolidBCStruct(
    bc::String,
    bb_lidxs::Union{Nothing,CuArray{Int32,1}},
    bb_dirs ::Union{Nothing,CuArray{Int8,1}},
    bb_ptr  ::Union{Nothing,CuArray{Int32,1}},
    nnz     ::Union{Nothing,Int32},
    bbOpp   ::Union{Nothing,CuArray{Int8,1}},
    rank::Int
)
    # normalize nnz
    nnz32 = nnz === nothing ? Int32(0) : Int32(nnz)

    if bc == "SBB"
        # Field order must match the old struct definition:
        # bb_lidxs, bb_dirs, bb_ptr, bbOpp, U_c_lat, V_c_lat, ω_lat, x_c_lat, y_c_lat, nnz, q_arr
        return SolidBCStruct(
            bb_lidxs,
            bb_dirs,
            bb_ptr,
            bbOpp,
            0f0,   # U_c_lat
            0f0,   # V_c_lat
            0f0,   # ω_lat
            0f0,   # x_c_lat
            0f0,   # y_c_lat
            nnz32, # nnz
            nothing # q_arr (unused for SBB)
        )

    elseif bc == "IBB"
        # q_arr stays nothing; IBB will compute q on-the-fly in the kernel
        return SolidBCStruct(
            bb_lidxs,
            bb_dirs,
            bb_ptr,
            bbOpp,
            0f0,   # U_c_lat
            0f0,   # V_c_lat
            0f0,   # ω_lat
            0f0,   # x_c_lat
            0f0,   # y_c_lat
            nnz32, # nnz
            nothing # q_arr
        )

    else
        error("\n\ninit: Unknown or unimplemented boundary condition: $bc.\n\n")
    end
end




const bc_dispatch = Dict{String, Function}(
    "SBB" => (f_old, f_new, rho, wvec, c_x, c_y, xlat, ylat, bcstruct, params, rank) -> SBB!(
        f_old, f_new, rho, wvec, c_x, c_y,
        bcstruct.bb_lidxs, bcstruct.bb_dirs, bcstruct.bbOpp, Int32(bcstruct.nnz),
        xlat, ylat,
        bcstruct.U_c_lat, bcstruct.V_c_lat, bcstruct.ω_lat,
        bcstruct.x_c_lat, bcstruct.y_c_lat,
        rank
    ),

    "IBB" => (f_old, f_new, rho, wvec, c_x, c_y, xlat, ylat, bcstruct, params, rank) -> IBB!(
        f_old, f_new, rho, wvec, c_x, c_y,
        bcstruct.bb_lidxs, bcstruct.bb_dirs, bcstruct.bbOpp, Int32(bcstruct.nnz),
        xlat, ylat, bcstruct.nbr_lli,
        bcstruct.U_c_lat, bcstruct.V_c_lat, bcstruct.ω_lat,
        bcstruct.x_c_lat, bcstruct.y_c_lat,
        bcstruct.shape_flag, bcstruct.R_lat,
        bcstruct.chord_lat, bcstruct.m, bcstruct.p, bcstruct.t,
        bcstruct.x_int, bcstruct.x_le_offset_lat,
        bcstruct.cθ_half, bcstruct.sθ_half,
        rank
    ),
)





function solid_BC!(
    f_old_gpu::CuArray{Float32,2},
    f_new_gpu::CuArray{Float32,2},
    rho_gpu::CuArray{Float32,1},
    wvec_gpu::CuArray{Float32,1},
    c_x_gpu::CuArray{Int8,1},
    c_y_gpu::CuArray{Int8,1},
    x_lli_lat::CuArray{Float32,1},
    y_lli_lat::CuArray{Float32,1},
    bcstruct::SolidBCStruct,
    params::LBMParams,
    bc::String,
    rank::Int
)
    if rank == 0
        return
    end

    if haskey(bc_dispatch, bc)
        bc_dispatch[bc](f_old_gpu, f_new_gpu, rho_gpu, wvec_gpu, c_x_gpu, c_y_gpu,
                        x_lli_lat, y_lli_lat, bcstruct, params, rank)
    else
        error("\n\nsolid_BC!: Unknown or unimplemented boundary condition: $bc.\n\n")
    end
end




# ---- BOUNDARY CONDITION FUNCTIONS ----

function SBB!_gpu(
    f_old    :: CuDeviceMatrix{Float32},   # destination (t+Δt)
    f_new    :: CuDeviceMatrix{Float32},   # post-collision f* at t
    rho      :: CuDeviceVector{Float32},   # local density
    wvec     :: CuDeviceVector{Float32},
    c_x      :: CuDeviceVector{Int8},
    c_y      :: CuDeviceVector{Int8},

    bb_lidxs :: CuDeviceVector{Int32},
    bb_dirs  :: CuDeviceVector{Int8},
    bbOpp    :: CuDeviceVector{Int8},
    nnz      :: Int32,

    x_lli_lat::CuDeviceVector{Float32},
    y_lli_lat::CuDeviceVector{Float32},

    U_c_lat::Float32,
    V_c_lat::Float32,
    ω_lat  ::Float32,
    x_c_lat::Float32,
    y_c_lat::Float32
)
    tid = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if !(1 <= tid <= nnz)
        return
    end

    i   = Int(@inbounds bb_lidxs[tid])     # fluid node column index
    dir = Int(@inbounds bb_dirs[tid])      # incoming direction hitting solid
    opp = Int(@inbounds bbOpp[dir])        # opposite direction

    cx = Float32(@inbounds c_x[dir])
    cy = Float32(@inbounds c_y[dir])    

    # wall velocity at this boundary-adjacent fluid node (lattice)
    xw = @inbounds x_lli_lat[i] - 0.5f0*cx
    yw = @inbounds y_lli_lat[i] - 0.5f0*cy

    rx = xw - x_c_lat
    ry = yw - y_c_lat

    uwx = U_c_lat - ω_lat * ry
    uwy = V_c_lat + ω_lat * rx

    ci_dot_uw = cx * uwx + cy * uwy
    rho_w     = @inbounds rho[i]
    wdir      = @inbounds wvec[dir]

    @inbounds f_old[opp, i] = f_new[dir, i] - 2f0 * wdir * rho_w * ci_dot_uw * INV_CS2
    return
end




function SBB!(
    f_old_gpu::CuArray{Float32,2},
    f_new_gpu::CuArray{Float32,2},
    rho_gpu::CuArray{Float32,1},
    wvec_gpu::CuArray{Float32,1},
    c_x_gpu::CuArray{Int8,1},
    c_y_gpu::CuArray{Int8,1},

    bb_lidxs_gpu::CuArray{Int32,1},
    bb_dirs_gpu ::CuArray{Int8,1},
    bbOpp_gpu   ::CuArray{Int8,1},
    nnz::Int32,

    x_lli_lat::CuArray{Float32,1},
    y_lli_lat::CuArray{Float32,1},

    U_c_lat::Float32,
    V_c_lat::Float32,
    ω_lat::Float32,
    x_c_lat::Float32,
    y_c_lat::Float32,
    rank::Int
)
    if rank == 0 || nnz == 0; return; end
    threads = 256
    blocks  = cld(Int(nnz), threads)
    @cuda threads=threads blocks=blocks SBB!_gpu(
        f_old_gpu, f_new_gpu, rho_gpu, wvec_gpu, c_x_gpu, c_y_gpu,
        bb_lidxs_gpu, bb_dirs_gpu, bbOpp_gpu, nnz,
        x_lli_lat, y_lli_lat,
        U_c_lat, V_c_lat, ω_lat, x_c_lat, y_c_lat
    )
    return
end




function IBB!_gpu(
    f_old     :: CuDeviceMatrix{Float32},     # (Q, N_local)   destination (t+Δt)
    f_new     :: CuDeviceMatrix{Float32},     # (Q, N_local)   post-collision f* at t
    rho_l     :: CuDeviceVector{Float32},     # local density
    wvec      :: CuDeviceVector{Float32},
    c_x       :: CuDeviceVector{Int8},
    c_y       :: CuDeviceVector{Int8},

    bb_lidxs  :: CuDeviceVector{Int32},       # nnz entries: boundary fluid nodes (b)
    bb_dirs   :: CuDeviceVector{Int8},        # direction i (points into solid)
    bbOpp     :: CuDeviceVector{Int8},
    nnz       :: Int32,

    # node positions (lattice units)
    x_lli_lat :: CuDeviceVector{Float32},
    y_lli_lat :: CuDeviceVector{Float32},

    # forward neighbor map: i_f for x_f = x_b + c_i  (Q × N_owned; -1 if off-domain)
    nbr_lli   :: CuDeviceMatrix{Int32},

    # half-time rigid-body kinematics (planar only)
    U_c_lat::Float32, V_c_lat::Float32, ω_lat::Float32,
    x_c_lat::Float32, y_c_lat::Float32,

    # geometry
    shape_flag::Int8,  R_lat::Float32,
    chord::Float32, m::Float32, p::Float32, t::Float32,
    x_int::Float32, x_le_offset::Float32,
    cθ::Float32, sθ::Float32
)
    tid = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if !(1 <= tid <= nnz); return; end

    i_b   = Int(@inbounds bb_lidxs[tid])     # boundary node column index
    dir   = Int(@inbounds bb_dirs[tid])      # incoming direction i
    i_opp = Int(@inbounds bbOpp[dir])        # opposite direction index

    # lattice link vector
    cx = Float32(@inbounds c_x[dir])
    cy = Float32(@inbounds c_y[dir])

    # positions: x_b and solid-side neighbor center
    xb = @inbounds x_lli_lat[i_b]
    yb = @inbounds y_lli_lat[i_b]
    xs = xb - cx;  ys = yb - cy; 

    # φ at both ends of the link (extruded geometry: φ ignores z)
    ϕb = phi_world_lat_3Dplanar(xb, yb, shape_flag, x_c_lat, y_c_lat,
                                R_lat, chord, m, p, t, x_int, x_le_offset, cθ, sθ)
    ϕs = phi_world_lat_3Dplanar(xs, ys, shape_flag, x_c_lat, y_c_lat,
                                R_lat, chord, m, p, t, x_int, x_le_offset, cθ, sθ)

    denom   = ϕb - ϕs
    no_brkt = (ϕb <= 0f0) | (ϕs >= 0f0) | (abs(denom) < 1f-8)

    if no_brkt
        # Fallback: moving SBB at half-link point
        xw = xb - 0.5f0*cx
        yw = yb - 0.5f0*cy
        (uwx, uwy) = uw_at_wall_lat_planar(xw, yw, x_c_lat, y_c_lat, U_c_lat, V_c_lat, ω_lat)
        ci_dot_uw  = cx*uwx + cy*uwy              # uw_z ≡ 0, cz term drops
        wdir       = @inbounds wvec[dir]
        rho_loc    = @inbounds rho_l[i_b]
        corr       = 2f0 * wdir * rho_loc * ci_dot_uw * INV_CS2
        @inbounds f_old[i_opp, i_b] = f_new[dir, i_b] - corr
        return
    end

    q = clamp(ϕb / denom, EPS_Q, 1f0 - EPS_Q)

    # wall intersection on the link (Δt=1): x_w = x_b - q c_i
    xw = xb - q*cx
    yw = yb - q*cy

    # moving-wall correction at x_w
    (uwx, uwy) = uw_at_wall_lat_planar(xw, yw, x_c_lat, y_c_lat, U_c_lat, V_c_lat, ω_lat)
    ci_dot_uw  = cx*uwx + cy*uwy                 # uw_z ≡ 0
    wdir       = @inbounds wvec[dir]
    rho_loc    = @inbounds rho_l[i_b]
    corr       = 2f0 * wdir * rho_loc * ci_dot_uw * INV_CS2   # subtract from reconstruction

    if q <= 0.5f0
        # forward fluid node: i_f for x_f = x_b + c_i
        i_f = Int(@inbounds nbr_lli[dir, i_b])
        if (i_f > 0) & (i_f != i_b)
            @inbounds f_old[i_opp, i_b] =
                2f0*q * f_new[dir, i_b] + (1f0 - 2f0*q) * f_new[dir, i_f] - corr
        else
            # missing or degenerate neighbor → switch to same-node branch
            @inbounds f_old[i_opp, i_b] =
                (0.5f0/q) * f_new[dir, i_b] + ((2f0*q - 1f0)/(2f0*q)) * f_new[i_opp, i_b] - corr
        end
    else
        @inbounds f_old[i_opp, i_b] =
            (0.5f0/q) * f_new[dir, i_b] + ((2f0*q - 1f0)/(2f0*q)) * f_new[i_opp, i_b] - corr
    end
    return
end




function IBB!(
    f_old_gpu::CuArray{Float32,2},
    f_new_gpu::CuArray{Float32,2},
    rho_l_gpu::CuArray{Float32,1},
    wvec_gpu::CuArray{Float32,1},
    c_x_gpu::CuArray{Int8,1},
    c_y_gpu::CuArray{Int8,1},

    bb_lidxs_gpu::CuArray{Int32,1},
    bb_dirs_gpu ::CuArray{Int8,1},
    bbOpp_gpu   ::CuArray{Int8,1},
    nnz::Int32,

    x_lli_lat::CuArray{Float32,1},
    y_lli_lat::CuArray{Float32,1},
    nbr_lli::CuArray{Int32,2},                 # (Q, N_owned)

    # half-time planar kinematics
    U_c_lat::Float32, V_c_lat::Float32, ω_lat::Float32,
    x_c_lat::Float32, y_c_lat::Float32,

    # geometry
    shape_flag::Int8,  R_lat::Float32,
    chord::Float32, m::Float32, p::Float32, t::Float32,
    x_int::Float32, x_le_offset::Float32,
    cθ::Float32, sθ::Float32,

    rank::Int
)
    if rank == 0 || nnz == 0; return; end
    threads = 256
    blocks  = cld(Int(nnz), threads)
    @cuda threads=threads blocks=blocks IBB!_gpu(
        f_old_gpu, f_new_gpu, rho_l_gpu, wvec_gpu, c_x_gpu, c_y_gpu,
        bb_lidxs_gpu, bb_dirs_gpu, bbOpp_gpu, nnz,
        x_lli_lat, y_lli_lat, nbr_lli,
        U_c_lat, V_c_lat, ω_lat, x_c_lat, y_c_lat,
        shape_flag, R_lat, chord, m, p, t, x_int, x_le_offset, cθ, sθ
    )
    return
end




end # module
